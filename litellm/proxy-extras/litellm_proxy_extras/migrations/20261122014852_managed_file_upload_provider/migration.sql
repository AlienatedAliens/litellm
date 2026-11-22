SET LOCAL lock_timeout = '10s';

ALTER TABLE "LiteLLM_ManagedFileTable" ADD COLUMN IF NOT EXISTS "upload_provider" TEXT;
