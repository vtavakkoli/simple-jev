# Security and responsible reporting

JEV Lab is experimental software. Hosted requests transmit state to TypeSafe; the public browser playground transmits state to Featherless. Local mode targets your chosen HF server. Review the provider and deployment configuration before submitting sensitive inputs.

## Credentials

Use `TYPESAFE_API_KEY` for the hosted client and Colab Secrets for notebooks. The CLI deliberately has no API-key argument, so keys do not enter shell history or process arguments. The local client never inherits the TypeSafe key. Do not commit `.env` files, request data, notebook secrets or private result logs.

The lightweight client refuses redirects, uses HTTPS outside loopback, performs no automatic retries, and does not fall back to another backend. These controls do not establish application-level safety: validate decisions and authorize actions in your application.

## Reporting a vulnerability

Use GitHub's **Security → Report a vulnerability** if private vulnerability reporting is enabled for this repository. If it is unavailable, open a minimal issue requesting a private contact without including secrets, exploit details or private data. A private contact channel and response-time commitment have not yet been established.

For leaked provider credentials, revoke them at the provider immediately; deleting a Git commit does not revoke a key. Provider-service vulnerabilities should be reported through the provider's own process.

## Scope and support

Report the commit, affected component, reproduction conditions and impact. There is currently no stable-release security support policy or security audit claim. Deployment authentication, rate limits, input retention and action permissions remain the operator's responsibility.
