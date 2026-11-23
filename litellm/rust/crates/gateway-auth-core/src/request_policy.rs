//! What the proxy takes before it reads a credential: where a request comes from and which route
//! it calls. No host is asked anything, so every host gets the same answer from the same settings.

use litellm_config::{GeneralSettings, RouteList};
use litellm_python_compat::lenient::Lenient;

use crate::error::RequestDenial;

#[derive(Clone, Debug, Default)]
pub struct Settings {
    /// [`general_settings.allowed_ips`](GeneralSettings::allowed_ips). Unset takes every address,
    /// and a value that is not a list of strings takes none.
    allowed_ips: Option<Box<[String]>>,
    /// [`general_settings.use_x_forwarded_for`](GeneralSettings::use_x_forwarded_for)
    use_x_forwarded_for: bool,
    /// [`general_settings.allowed_routes`](GeneralSettings::allowed_routes). Unset takes every
    /// route, and a value that is not a list of strings takes none.
    allowed_routes: Option<Box<[String]>>,
}

impl Settings {
    pub fn new(general: &GeneralSettings) -> Self {
        Self {
            allowed_ips: general
                .allowed_ips
                .as_ref()
                .map(|allowed| Lenient::items(allowed).into()),
            use_x_forwarded_for: general.use_x_forwarded_for == Some(true),
            allowed_routes: general
                .allowed_routes
                .as_ref()
                .map(|allowed| RouteList::items(allowed).into()),
        }
    }

    /// `RouteChecks.is_allowed_on_proxy(route)`: the route as written, with no pattern or group.
    pub fn permits_route(&self, route: &str) -> bool {
        self.allowed_routes
            .as_ref()
            .is_none_or(|allowed| allowed.iter().any(|entry| entry == route))
    }

    fn permits_address(&self, address: &str) -> bool {
        self.allowed_ips
            .as_ref()
            .is_none_or(|allowed| allowed.iter().any(|entry| entry == address))
    }
}

/// Where a request came from, as the host's server reports it.
#[derive(Clone, Copy, Debug, Default)]
pub struct Origin<'a> {
    /// `request.client.host`
    pub peer: Option<&'a str>,
    /// The `x-forwarded-for` header as the request sent it.
    pub forwarded_for: Option<&'a str>,
}

impl<'a> Origin<'a> {
    /// `get_request_ip_address`: the forwarded header whole when the operator reads it and the
    /// request sent one, else the peer.
    fn address(&self, settings: &Settings) -> &'a str {
        match self.forwarded_for {
            Some(forwarded) if settings.use_x_forwarded_for => forwarded,
            _ => self.peer.unwrap_or_default(),
        }
    }
}

/// The address, then the route, as `pre_db_read_auth_checks` orders them.
pub(crate) fn check(
    route: &str,
    origin: Origin<'_>,
    settings: &Settings,
) -> Result<(), RequestDenial> {
    let address = origin.address(settings);
    if !settings.permits_address(address) {
        return Err(RequestDenial::AddressNotAllowed {
            address: address.to_owned(),
        });
    }
    if !settings.permits_route(route) {
        return Err(RequestDenial::RouteNotAllowed {
            route: route.to_owned(),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;
    use crate::PublicDenial;

    const ROUTE: &str = "/v1/messages";

    fn list(entries: &[&str]) -> Lenient<Box<[String]>> {
        Lenient::Valid(entries.iter().map(|entry| (*entry).to_owned()).collect())
    }

    fn addresses(allowed: Option<Lenient<Box<[String]>>>, use_x_forwarded_for: bool) -> Settings {
        Settings::new(&GeneralSettings {
            allowed_ips: allowed,
            use_x_forwarded_for: Some(use_x_forwarded_for),
            ..GeneralSettings::default()
        })
    }

    fn routes(allowed: Option<RouteList>) -> Settings {
        Settings::new(&GeneralSettings {
            allowed_routes: allowed,
            ..GeneralSettings::default()
        })
    }

    fn refused_address(address: &str) -> Result<(), RequestDenial> {
        Err(RequestDenial::AddressNotAllowed {
            address: address.to_owned(),
        })
    }

    #[rstest]
    #[case::unset_takes_every_address(None, false, Some("10.0.0.9"), None, Ok(()))]
    #[case::a_listed_peer(Some(list(&["10.0.0.1", "10.0.0.2"])), false, Some("10.0.0.2"), None, Ok(()))]
    #[case::an_unlisted_peer(Some(list(&["10.0.0.1"])), false, Some("10.0.0.9"), None, refused_address("10.0.0.9"))]
    #[case::an_empty_list_takes_none(Some(list(&[])), false, Some("10.0.0.1"), None, refused_address("10.0.0.1"))]
    #[case::a_malformed_list_takes_none(
        Some(Lenient::Malformed),
        false,
        Some("10.0.0.1"),
        None,
        refused_address("10.0.0.1")
    )]
    #[case::no_peer_is_the_empty_address(Some(list(&["10.0.0.1"])), false, None, None, refused_address(""))]
    #[case::an_unread_forwarded_header_changes_nothing(Some(list(&["10.0.0.1"])), false, Some("10.0.0.1"), Some("10.0.0.9"), Ok(()))]
    #[case::a_read_forwarded_header_replaces_the_peer(Some(list(&["10.0.0.1"])), true, Some("10.0.0.1"), Some("10.0.0.9"), refused_address("10.0.0.9"))]
    #[case::a_read_forwarded_header_that_is_listed(Some(list(&["10.0.0.9"])), true, Some("10.0.0.1"), Some("10.0.0.9"), Ok(()))]
    #[case::the_forwarded_header_is_compared_whole(Some(list(&["10.0.0.9"])), true, Some("10.0.0.1"), Some("10.0.0.9, 10.0.0.1"), refused_address("10.0.0.9, 10.0.0.1"))]
    #[case::no_forwarded_header_falls_back_to_the_peer(Some(list(&["10.0.0.1"])), true, Some("10.0.0.1"), None, Ok(()))]
    fn the_address_is_checked_against_allowed_ips(
        #[case] allowed: Option<Lenient<Box<[String]>>>,
        #[case] use_x_forwarded_for: bool,
        #[case] peer: Option<&str>,
        #[case] forwarded_for: Option<&str>,
        #[case] expected: Result<(), RequestDenial>,
    ) {
        let origin = Origin {
            peer,
            forwarded_for,
        };

        assert_eq!(
            check(ROUTE, origin, &addresses(allowed, use_x_forwarded_for)),
            expected
        );
    }

    #[rstest]
    #[case::unset_takes_every_route(None, true)]
    #[case::a_listed_route(Some(list(&["/health", ROUTE])), true)]
    #[case::an_unlisted_route(Some(list(&["/health"])), false)]
    #[case::a_prefix_is_not_a_match(Some(list(&["/v1"])), false)]
    #[case::a_pattern_is_not_read(Some(list(&["/v1/*"])), false)]
    #[case::an_empty_list_takes_none(Some(list(&[])), false)]
    #[case::a_malformed_list_takes_none(Some(Lenient::Malformed), false)]
    fn the_route_is_checked_against_allowed_routes(
        #[case] allowed: Option<RouteList>,
        #[case] permitted: bool,
    ) {
        let settings = routes(allowed);
        let expected = if permitted {
            Ok(())
        } else {
            Err(RequestDenial::RouteNotAllowed {
                route: ROUTE.to_owned(),
            })
        };

        assert_eq!(settings.permits_route(ROUTE), permitted);
        assert_eq!(check(ROUTE, Origin::default(), &settings), expected);
    }

    #[test]
    fn the_address_is_refused_before_the_route() {
        let settings = Settings::new(&GeneralSettings {
            allowed_ips: Some(list(&["10.0.0.1"])),
            allowed_routes: Some(list(&["/health"])),
            ..GeneralSettings::default()
        });
        let origin = Origin {
            peer: Some("10.0.0.9"),
            forwarded_for: None,
        };

        assert_eq!(check(ROUTE, origin, &settings), refused_address("10.0.0.9"));
    }

    #[rstest]
    #[case::address(
        RequestDenial::AddressNotAllowed { address: "10.0.0.9".to_owned() },
        "Access forbidden: IP address 10.0.0.9 not allowed."
    )]
    #[case::route(
        RequestDenial::RouteNotAllowed { route: ROUTE.to_owned() },
        "Access forbidden: Route /v1/messages not allowed"
    )]
    fn a_denial_is_the_403_the_proxy_raised(#[case] denial: RequestDenial, #[case] detail: &str) {
        assert_eq!(
            denial.public(),
            PublicDenial::Http {
                status: 403,
                detail: detail.to_owned(),
            }
        );
    }
}
