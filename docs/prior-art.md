# Prior art

rampgate builds on well-known ideas:

- **Limits learned from usage history:** Google's Autopilot and the Kubernetes Vertical Pod Autoscaler.
- **Reclaiming reserved but unused resources:** Borg.
- **Sharing GPUs between deep-learning jobs:** research systems such as Gandiva, Salus and AntMan.

Its contribution is the **combination** of serialized ramp-up (probation), observed-peak reservation and OOM feedback, packaged as a small **embeddable library** with zero runtime dependencies and a deterministic simulator. It is not a cluster manager and does not replace one. It decides only *when* a job may start on a device the caller already chose.
