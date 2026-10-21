# DiagEvidence security policy

Version 0.1.1 is a research preview. The intended maintenance line is 0.1.x. Once
releases exist, fixes will be delivered in the latest version of that line.
This project does not provide a service-level agreement or a deployment
hardening certification.

Use GitHub's **Report a vulnerability** option in this repository
for a suspected vulnerability if that option is enabled. Otherwise contact the maintainer using a
private method listed on their GitHub profile. If neither is available, open a
minimal issue requesting a private reporting channel. Do not include exploit
details, credentials, real signals or identifying logs in that public request.

A useful private report includes the affected version, a minimal synthetic
reproduction, expected and observed behavior, and the potential impact. The
maintainer will evaluate the report and coordinate a fix and disclosure when
possible. No response deadline is promised.

Treat external diagnostic and retrieval services as separate trust boundaries.
Their operators control their authentication and data handling. The package's
sanitizer reduces accidental disclosure in local artifacts; it is not a full
data-loss-prevention system. Do not run untrusted model files or distribute
third-party checkpoints without checking their license and provenance.
