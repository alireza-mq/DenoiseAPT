# Security and deployment scope

DenoiseAPT is a local research demonstration, not a hardened network service.
It has no authentication, authorization, multi-user isolation, or persistent
server-side session store.

- Keep the default loopback binding (`127.0.0.1`).
- Do not expose the server directly to the public Internet.
- Treat uploaded CSV values and exported JSON as research data.
- Review upstream data terms before downloading optional datasets.
- Do not use the configured scorer checks as safety, clinical, or production
  certification.

Please report security issues privately to the repository owners rather than
posting sensitive details in a public issue.
