SET LOCAL lock_timeout = '10s';

ALTER TABLE "LiteLLM_ManagedFileTable" ADD COLUMN "upload_provider" TEXT;
