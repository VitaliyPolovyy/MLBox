# MLOps Security Architecture

**Document Version:** 2.0  
**Last Updated:** November 21, 2025  
**Project:** MLBox Ray Application

---

## 1. System Overview

### Components
- **MLBox Service**: Ray application (port 8000)
- **Production ERP**: Single ERP with UI selector (Test/Prod buttons)
- **Docker Registry**: Container image repository (port 5000)
- **End Users**: Browser-based access via ERP

### Core Security Principles
- Defense in depth (multiple security layers)
- Least privilege (minimum required access)
- Zero trust (verify everything)
- Fail securely (safe defaults on errors)
- Assume breach (design for containment)

---

## 2. Architecture Design

### Target Architecture (3-Tier)

```
Dev Server (isolated)
    ↓ (push images)
Test/Staging Server
    - Docker Registry
    - Staging Service
    ↓ (Prod pulls images)
Prod Server
    - Production Service
```

### Environment Characteristics

**Dev Server**
- Purpose: Development, experimentation, image building
- Internet: Full access (PyPI, HuggingFace, GitHub, etc.)
- Risk Level: HIGH (most likely to be compromised)
- Isolation: Cannot access Test/Prod services or ERP
- Data: Synthetic/mock data only

**Test/Staging Server**
- Purpose: Pre-production validation, hosts registry
- Internet: Restricted (whitelisted domains only)
- Risk Level: SAME AS PRODUCTION
- Security: Production-grade (handles real production data)
- Data: Real production data (via ERP)
- Dual Role: Registry + staging service

**Prod Server**
- Purpose: Production workload
- Internet: Highly restricted (whitelisted domains only)
- Risk Level: PRODUCTION
- Security: Maximum controls
- Data: Real production data

### Key Architectural Insight

**Test and Prod are twins with different image versions:**
- Identical network security
- Identical ERP access
- Identical data access
- Different: Docker image version only (Test runs beta, Prod runs stable)

**"Test" is a misnomer** - it's actually a **Staging/Beta/Preview environment** with production-grade security.

---

## 3. Registry Placement Strategy

### The Core Problem

Docker registry must exist somewhere. Registry placement determines security posture.

**Key constraint:** Dev server has full internet access (high risk).

### Option Analysis

#### Option A: Registry on Dev (❌ Not Recommended)
```
Risk: Prod pulls from high-risk environment
Problem: If Dev compromised, registry itself compromised
```

#### Option B: Registry on Prod (⚠️ Temporary Acceptable)
```
Risk: Dev pushes to production server
Problem: High-risk environment connects to Prod
Mitigation: Strong security controls required
```

#### Option C: Registry on Test/Staging (✅ Recommended)
```
Benefit: No direct Dev→Prod connection
Benefit: Registry in secure zone (same as Prod)
Benefit: Logical deployment flow
```

#### Option D: Separate Registry Server (✅✅ Most Secure)
```
Benefit: Maximum isolation
Benefit: Dedicated resources
Drawback: Additional infrastructure
```

### Selected Strategy

**Current (2 servers):** Registry on Prod with strong mitigations  
**Future (3 servers):** Registry on Test/Staging server  
**Ideal (4 servers):** Dedicated registry server

---

## 4. Network Security

### Firewall Philosophy

**One-way connection initiation:**
- Dev can initiate connections to Test/Staging
- Test/Staging and Prod cannot initiate connections to Dev
- Prevents pivot attacks if Prod compromised

**Stateful firewalls:**
- Allow NEW connections only where specified
- Response traffic automatically allowed (established connections)
- Bidirectional data flow on Dev-initiated connections

### Network Flows

**Dev Server**
```
Outbound:
- → Internet (full access for development)
- → Test/Staging:5000 (registry push)

Blocked:
- ✗ Test/Staging:8000 (cannot access services)
- ✗ Test/Staging:22 (cannot SSH)
- ✗ Prod:* (no production access)
- ✗ ERP (no ERP access)
```

**Test/Staging Server**
```
Inbound:
- ← Dev:5000 (registry push)
- ← Prod:5000 (registry pull)
- ← ERP:8000 (staging service calls)
- ← QA/Users:8000 (testing access)

Outbound:
- → HuggingFace (model downloads)
- → ERP (callbacks, optional)

Blocked:
- ✗ Dev:* (cannot connect back to Dev)
- ✗ General internet
```

**Prod Server**
```
Inbound:
- ← ERP:8000 (production service calls)
- ← End Users:8000 (browser access)

Outbound:
- → Test/Staging:5000 (registry pull)
- → HuggingFace (model downloads)
- → ERP (callbacks, optional)

Blocked:
- ✗ Dev:* (no Dev access)
- ✗ General internet
```

### Critical Security Rule

**Test/Staging and Prod can NEVER initiate connections to Dev.**

This prevents:
- Pivot attacks if Prod compromised
- Lateral movement from secure to less secure zones
- Attack containment (Dev compromise doesn't automatically spread)

---

## 5. Image Security

### The Threat Model

**Primary risk:** Dev server compromise
- Dev has full internet access
- Supply chain attacks (malicious packages)
- Phishing, social engineering
- If Dev compromised, attacker can push images

**Why this matters:**
- Compromised Dev → malicious images
- Malicious images → production breach
- Need verification before deployment

### Security Layers

**Layer 1: Image Signing**
- Cryptographic proof of authenticity
- Private key for signing (kept off Dev, in CI/CD)
- Public key for verification (on Test/Prod)
- Unsigned images rejected

**Layer 2: Vulnerability Scanning**
- Scan for known security issues
- Block images with critical vulnerabilities
- Automated scanning before push
- Re-scanning on Prod before deployment

**Layer 3: Registry Authentication**
- Separate credentials per environment
- Dev: write-only access (can push)
- Prod: read-only access (can pull)
- TLS/HTTPS encryption mandatory

**Layer 4: Verification Before Deployment**
- Verify signature on Prod before pulling
- Scan for vulnerabilities on Prod
- Manual approval gate (optional)
- Automated rollback on failure

**Layer 5: Audit Logging**
- Log all registry operations
- Monitor push/pull activity
- Alert on anomalies
- Forensics capability

### Defense in Depth

**If Dev compromised and attacker pushes malicious image:**
1. ❌ Signing verification fails (attacker doesn't have key)
2. ❌ Vulnerability scan detects malware
3. ❌ Monitoring alerts on suspicious activity
4. ❌ Manual approval rejects deployment
5. ✅ Production protected

**Multiple independent controls** - not relying on single defense.

---

## 6. Deployment Workflow

### Standard Flow

```
1. Developer commits code to Git
2. CI/CD pipeline builds image
3. Automated vulnerability scan
4. Image signing (CI/CD has key, not Dev)
5. Push to registry (Test/Staging server)
6. Deploy to Staging automatically
7. QA/users test via ERP "Test" button
8. Validation period (24-48 hours)
9. Manual approval to promote
10. Deploy to Production
11. Users access via ERP "Prod" button
```

### Key Principles

**Automated deployment** (reduces human error)  
**Immutable infrastructure** (don't patch, replace)  
**Gradual rollout** (staging first, then prod)  
**Quick rollback** (keep previous version ready)

---

## 7. Risk Analysis

### Attack Scenarios

#### Scenario 1: Dev Server Compromised (Most Likely)

**Without mitigations:**
- Attacker pushes malicious images
- Production deploys and executes malware
- Production data compromised

**With mitigations:**
- Attacker cannot sign images (key not on Dev)
- Vulnerability scanning blocks malware
- Signature verification fails on Prod
- Deployment blocked, alerts triggered

**Containment:**
- Attacker limited to Dev environment
- Cannot directly access Test/Prod services
- Cannot SSH to production servers
- Attack detected and contained

#### Scenario 2: Prod Server Compromised (Less Likely)

**Impact:**
- Critical incident (production breach)
- Production data at risk
- Immediate response required

**Containment:**
- Prod cannot pivot to Dev (firewall blocks)
- Attack limited to production zone
- Cannot compromise Dev or registry
- Lateral movement prevented

#### Scenario 3: Registry Compromise

**If registry on Prod:**
- Registry in secure zone
- Harder to compromise
- Signature verification still protects deployments

**If registry on Dev:**
- Registry in high-risk zone
- More likely to be compromised
- Source of truth untrusted

**Why registry on Test/Staging is best:**
- Registry in secure zone
- No direct Dev→Prod connection
- Defense in depth maintained

---

## 8. Current State vs Future State

### Current State (2 Servers)

```
Dev → Prod (registry + service)
```

**Challenges:**
- Direct Dev→Prod connection exists
- Registry and service on same server
- Resource sharing
- Higher risk

**Required mitigations:**
- Image signing mandatory
- Vulnerability scanning mandatory
- Strict firewall (only port 5000)
- Enhanced monitoring
- Manual approval gates

**Risk Level:** ⚠️ Acceptable with strong controls, but not ideal

---

### Future State (3 Servers - Recommended)

```
Dev → Test/Staging (registry + service) → Prod
```

**Benefits:**
- No direct Dev→Prod connection
- Registry in secure zone
- Staging validation before prod
- Clear separation of concerns
- Industry best practice

**Risk Level:** ✅ Good security posture

---

### Ideal State (4 Servers)

```
Dev → Registry Server (dedicated) ← Test/Staging
                                    ↓
                                   Prod
```

**Benefits:**
- Maximum isolation
- Dedicated registry resources
- Optimal security
- Scalable architecture

**Risk Level:** ✅✅ Excellent security

---

## 9. ERP Integration

### User Workflow

**Production ERP Interface:**
```
[Preview/Beta Service v1.5] [Stable Service v1.4]
           ↓                         ↓
    Test/Staging:8000           Prod:8000
    (new features)              (stable version)
```

### Use Cases

**Test/Staging Service:**
- QA team validation
- Power users / early adopters
- New feature testing
- ML model accuracy validation
- Performance testing
- Integration testing

**Production Service:**
- General user population
- Stable, validated version
- Known-good performance
- Recommended default

### Deployment Strategy

**Gradual rollout:**
1. Deploy v1.5 to Staging
2. 10% users test via "Preview" button
3. Collect feedback, monitor metrics
4. If stable, promote to Production
5. All users migrate to v1.5

**A/B testing:**
- Same data processed by both versions
- Compare results and performance
- Validate ML model improvements
- Catch regressions before full rollout

---

## 10. Monitoring & Response

### What to Monitor

**Registry Activity:**
- Push/pull operations
- Authentication attempts
- Unusual image sizes
- Push times (off-hours suspicious)
- Source IPs

**Deployment Events:**
- Image verification results
- Vulnerability scan results
- Deployment successes/failures
- Rollback events

**System Health:**
- Service availability
- Resource utilization
- Error rates
- Response times

### Alert Triggers

**Critical (Immediate Response):**
- Signature verification failure
- Critical vulnerabilities detected
- Unauthorized registry access
- Deployment to Prod without approval

**Warning (Review Within Hours):**
- Push from unexpected IP
- Push outside business hours
- Multiple failed authentications
- Unusual image characteristics

**Info (Daily Review):**
- Normal deployments
- Routine scans
- Regular pull operations

### Incident Response

**Detection → Containment → Investigation → Recovery → Lessons Learned**

Quick response capability:
- Automated rollback procedures
- Network isolation capability
- Image quarantine process
- Credential rotation workflows
- Forensics log preservation

---

## 11. Security Best Practices

### Network
- One-way connection enforcement (Dev cannot receive from Prod)
- Minimal port exposure (only 5000, 8000)
- IP whitelisting where possible
- Stateful firewall rules
- Network segmentation by security level

### Images
- Always sign images (mandatory)
- Scan before deployment (block vulnerable images)
- Use immutable digests, not mutable tags
- Regular vulnerability rescans
- Image provenance tracking

### Access
- Separate credentials per environment
- Read-only for production pulls
- Write-only for dev pushes (if possible)
- Automated deployment only (no manual pushes)
- Principle of least privilege

### Operations
- Immutable infrastructure (replace, don't patch)
- Automated rollback capability
- Comprehensive audit logging
- Regular security reviews
- Continuous improvement

---

## 12. Implementation Priorities

### Phase 1: Foundation (Immediate)

**If 2 servers (Dev + Prod):**
- Set up registry on Prod with authentication
- Implement image signing (Cosign)
- Configure strict firewall rules
- Basic monitoring and alerting

**Risk:** Acceptable with mitigations

---

### Phase 2: Enhanced Security (Weeks 2-4)

**Add security layers:**
- Vulnerability scanning (Trivy)
- Automated verification in deployment
- Enhanced monitoring and alerts
- Manual approval gates

**Risk:** Good with multiple controls

---

### Phase 3: Proper Architecture (Month 2-3)

**Add Test/Staging server:**
- Request infrastructure
- Configure networking
- Migrate registry to Test/Staging
- Set up staging service
- Eliminate Dev→Prod connection

**Risk:** Industry standard, recommended posture

---

### Phase 4: Optimization (Ongoing)

**Continuous improvement:**
- Advanced monitoring (anomaly detection)
- Automated security testing
- Regular penetration testing
- Team security training
- Process refinement

---

## 13. Key Decisions & Rationale

### Why 3-Tier Architecture?

**Need staging validation:**
- Test new ML models before production
- Catch bugs before users see them
- ERP integration testing
- Gradual feature rollout

**Security isolation:**
- Dev isolated (high risk, full internet)
- Test/Staging and Prod in secure zone
- No direct Dev→Prod connection

### Why Registry on Test/Staging?

**Options considered:**
- ❌ Registry on Dev: Registry in high-risk zone
- ⚠️ Registry on Prod: Dev→Prod connection required
- ✅ Registry on Test/Staging: No Dev→Prod, secure zone
- ✅✅ Separate registry: Best but more infrastructure

**Selected:** Test/Staging (balances security and complexity)

### Why Single Production ERP?

**Constraint:** No separate test ERP available

**Implication:** Test/Staging must have production-grade security
- Processes real production data
- Connected to production ERP
- Same security controls as Prod

**Not a problem:** Test/Staging designed for this

### Why Image Signing is Critical?

**Dev is high-risk:**
- Full internet access
- Most likely to be compromised
- Attacker can push images

**Signing provides:**
- Cryptographic proof of authenticity
- Tamper detection
- Key not on Dev (in CI/CD)
- Even if Dev compromised, attacker cannot sign

**This is the most important control.**

---

## 14. Common Pitfalls to Avoid

### Security Anti-Patterns

❌ Registry on Dev server (high risk)  
❌ No image signing (no authenticity proof)  
❌ Bidirectional Dev↔Prod access  
❌ Same credentials for all environments  
❌ No vulnerability scanning  
❌ Manual deployment processes  
❌ No monitoring or alerting  
❌ Production data on Dev  
❌ Trusting tags instead of digests  
❌ SSH access from Dev to Prod  

### Correct Patterns

✅ Registry in secure zone (Test/Staging or separate)  
✅ Mandatory image signing  
✅ One-way connection (Dev can push, Prod cannot respond)  
✅ Separate credentials (Dev write, Prod read)  
✅ Automated vulnerability scanning  
✅ CI/CD automation  
✅ Comprehensive monitoring  
✅ Mock/synthetic data on Dev only  
✅ Immutable image digests  
✅ No direct Dev→Prod access  

---

## 15. Quick Reference

### Security Posture by Configuration

| Configuration | Security Level | Notes |
|---------------|----------------|-------|
| Dev → Prod (registry on Prod) | ⚠️ 6/10 | Temporary, needs strong mitigations |
| Dev → Prod (registry on Dev) | ⚠️ 5/10 | Not recommended |
| Dev → Test (registry) → Prod | ✅ 9/10 | Recommended, industry standard |
| Dev → Registry → Test → Prod | ✅✅ 10/10 | Ideal, maximum isolation |

### Required Security Controls

**Minimum (Must Have):**
- Image signing
- Vulnerability scanning
- Registry authentication
- TLS/HTTPS
- Firewall rules (one-way)

**Recommended (Should Have):**
- Automated CI/CD
- Manual approval gates
- Monitoring and alerting
- Audit logging
- Regular security reviews

**Advanced (Nice to Have):**
- Anomaly detection
- Penetration testing
- Image provenance (SLSA)
- Runtime admission control
- Zero-trust networking

---

## 16. Summary

### Current Challenge
- 2 servers (Dev + Prod)
- Need registry somewhere
- Dev has full internet (high risk)
- Production system with real data

### Solution
**Now:** Registry on Prod with strong mitigations  
**Future:** Add Test/Staging, move registry there  
**Goal:** No direct Dev→Prod connection

### Critical Controls
1. **Image signing** (prevents unauthorized images)
2. **Vulnerability scanning** (catches known malware)
3. **One-way connections** (prevents pivot attacks)
4. **Verification before deployment** (trust but verify)
5. **Monitoring and alerting** (detect anomalies)

### Success Criteria
- ✅ Dev isolated from production systems
- ✅ All images signed and verified
- ✅ Vulnerabilities caught before deployment
- ✅ Prod can validate in staging before rollout
- ✅ Quick rollback capability
- ✅ Comprehensive audit trail

**This architecture balances security, practicality, and operational efficiency for MLOps deployments.**

---

## Appendix: Glossary

**Dev/Development:** Environment for building and testing code  
**Test/Staging:** Pre-production validation environment (production-grade security)  
**Prod/Production:** Live environment serving real users  
**Registry:** Storage for Docker container images  
**Image signing:** Cryptographic verification of image authenticity  
**Vulnerability scanning:** Automated detection of known security issues  
**One-way connection:** Connection initiated from one side only, not bidirectional  
**Stateful firewall:** Firewall that tracks connection state  
**Defense in depth:** Multiple independent security layers  
**Least privilege:** Minimum required access only  
**Zero trust:** Verify everything, trust nothing  
**Immutable infrastructure:** Replace instead of modify  
**CI/CD:** Continuous Integration/Continuous Deployment automation  
