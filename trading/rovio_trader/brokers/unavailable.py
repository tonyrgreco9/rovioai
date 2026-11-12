"""Integrations that are reported accurately instead of being faked."""
from __future__ import annotations

IBKR_STATUS = {
    "broker": "Interactive Brokers (futures: MES, MNQ)",
    "paper_adapter_implemented": False,
    "summary": "Not implemented in this release. Futures run in replay and backtest only.",
    "why": "The TWS API is a socket protocol served by a locally running Trader Workstation or IB Gateway. It could not be reached, exercised or verified "
           "from the environment this was built in, and shipping an order-routing adapter that has never talked to its counterpart would be worse than not shipping one.",
    "what_it_needs": [
        "TWS or IB Gateway running on your machine, logged in to a PAPER account, with 'Enable ActiveX and Socket Clients' on (paper ports: TWS 7497, Gateway 4002)",
        "Futures trading permission and a CME real-time market-data subscription on the account (paper accounts share the live account's subscriptions)",
        "An adapter implementing brokers/base.py (contract details for multiplier / tick / expiry, bracket orders with an orderRef as the stable client reference)",
    ],
}

FIDELITY_STATUS = {
    "broker": "Fidelity (preferred brokerage)",
    "order_entry_available": False,
    "account_data_available": "Only through Fidelity Access, Fidelity's consented data-sharing channel for approved third-party aggregators. It is read-only.",
    "checked": "2026-11-12 (public sources; Fidelity publishes no retail trading API documentation to check against)",
    "summary": "No officially supported way to place orders in a retail Fidelity account from your own software was found. Fidelity execution stays disabled.",
    "findings": [
        "Retail brokerage accounts: no public order-entry API. Active Trader Pro / Fidelity Trader+ expose no automation interface.",
        "Fidelity Access: account-data sharing with approved aggregators via customer consent; it does not place trades.",
        "Order-capable connectivity (Integration Xchange / Wealthscape APIs, FIX) is offered by Fidelity Institutional to its custody-and-clearing clients "
        "(RIAs, broker-dealers, family offices), not to individual retail accounts.",
        "Browser-automation libraries exist; they log in with your password, conflict with Fidelity's terms on automated access, and are not used here.",
    ],
    "what_would_change_this": [
        "Confirm your exact account type with Fidelity. If it is held through an advisor or firm that custodies with Fidelity Institutional, ask that firm whether "
        "API or FIX order entry is available to you and under what agreement.",
        "Written confirmation from Fidelity of an approved order-entry integration for your account and asset types, plus its official documentation.",
        "Only then: an adapter implementing brokers/base.py, verified on whatever non-production environment Fidelity provides.",
    ],
    "your_account_type": "unknown to this software: it has never connected to Fidelity and never asks for Fidelity credentials",
}
