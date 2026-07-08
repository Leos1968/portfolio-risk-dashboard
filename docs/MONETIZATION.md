# Monetization Blueprint — Paid Analytics API

How to take the analytics engine in this repo from a portfolio piece to a
billable product: containerize it, gate it with API keys, and meter it with
Stripe Billing.

## 1. Architecture

```
                        ┌─────────────────────────────┐
  Customer ──X-API-Key──▶  FastAPI (backend/main.py)  │──▶ analytics.py
                        │  · key validation           │
                        │  · usage metering           │
                        └──────────────┬──────────────┘
                                       │ usage records
        Stripe Checkout ──webhook──▶ key issuance / revocation
        Stripe Billing  ◀──────────── metered usage reports
```

The quantitative engine (`analytics.py`) stays pure; monetization is a thin
layer around the existing `require_api_key` dependency in `main.py`.

## 2. Containerize (already done)

`Dockerfile` + `docker-compose.yml` in the repo root run both services. For a
paid deployment you ship only the `api` service:

```bash
docker build -t portfolio-risk-api .
docker run -p 8000:8000 -e PORTFOLIO_API_KEYS="<issued-keys>" portfolio-risk-api
```

Deploy anywhere containers run (Fly.io, Cloud Run, ECS). Put TLS in front
(the platform usually provides it) and set `allow_origins` in `main.py` to
your dashboard's domain.

## 3. Key issuance with Stripe Checkout

Flow:

1. Customer buys a plan via a Stripe **Checkout Session** (subscription mode).
2. Stripe fires a `checkout.session.completed` **webhook**.
3. Your webhook handler generates an API key, stores it (Redis/Postgres)
   against the Stripe customer ID, and emails it to the customer.
4. `customer.subscription.deleted` webhook → revoke the key.

Minimal webhook handler (add to `backend/main.py` or a separate service):

```python
import secrets, stripe
from fastapi import Request, Header, HTTPException

stripe.api_key = os.environ["STRIPE_SECRET_KEY"]
WEBHOOK_SECRET = os.environ["STRIPE_WEBHOOK_SECRET"]

@app.post("/stripe/webhook", include_in_schema=False)
async def stripe_webhook(request: Request, stripe_signature: str = Header(None)):
    event = stripe.Webhook.construct_event(
        await request.body(), stripe_signature, WEBHOOK_SECRET
    )
    if event["type"] == "checkout.session.completed":
        customer_id = event["data"]["object"]["customer"]
        api_key = f"prk_{secrets.token_urlsafe(24)}"
        key_store.save(api_key, customer_id)          # Redis / Postgres
        email_key_to_customer(customer_id, api_key)
    elif event["type"] == "customer.subscription.deleted":
        key_store.revoke_by_customer(event["data"]["object"]["customer"])
    return {"received": True}
```

Swap the env-var key list in `require_api_key` for a `key_store.is_valid(key)`
lookup once you have real customers.

## 4. Metered billing

For usage-based pricing, report each analytics call to Stripe:

```python
# inside the analytics endpoints, after a successful computation
stripe.billing.MeterEvent.create(
    event_name="portfolio_analysis",
    payload={"stripe_customer_id": customer_id, "value": 1},
)
```

Create a **Meter** in the Stripe dashboard and attach it to a metered price
(e.g. $0.01 per analysis after 1,000 free calls/month). Stripe aggregates the
events and invoices automatically.

## 5. Suggested pricing tiers

| Plan       | Price       | Included                                  |
|------------|-------------|-------------------------------------------|
| Free       | $0          | 100 analyses/mo, community support        |
| Pro        | $29/mo      | 10k analyses/mo, priority support          |
| Enterprise | Custom      | Unlimited, SLA, on-prem container license |

## 6. Hardening checklist before charging money

- [ ] Rate limiting (e.g. `slowapi`) per key
- [ ] Key hashing at rest; never log raw keys
- [ ] Structured logging + request IDs
- [ ] `pytest` suite for `analytics.py` (pure functions — trivial to test)
- [ ] CI (GitHub Actions): lint (`ruff`), tests, Docker build
- [ ] Status page + uptime monitoring
