# Canonical Insurance Fraud Data Model

## Purpose

This document defines the internal data model used by the Fraud Intelligence Platform.

External datasets must be transformed into this canonical representation before they are used by downstream components.

## Core Entities

### Customer

Represents the policy holder or claimant.

Fields:

- customer_id
- name
- date_of_birth
- phone
- email
- address_id

### Policy

Represents an insurance policy.

Fields:

- policy_id
- customer_id
- policy_type
- policy_start_date
- policy_end_date
- premium
- deductible
- coverage_amount
- status

### Claim

Represents an insurance claim.

Fields:

- claim_id
- policy_id
- customer_id
- claim_date
- incident_date
- claim_type
- claim_description
- claimed_amount
- approved_amount
- status
- fraud_label

### Provider

Represents a repair shop, medical provider, contractor, merchant, or other service provider.

Fields:

- provider_id
- provider_type
- provider_name
- address_id

### Transaction

Represents a financial or claim-related transaction.

Fields:

- transaction_id
- customer_id
- claim_id
- provider_id
- transaction_date
- amount
- transaction_type

### Address

Represents a reusable geographic entity.

Fields:

- address_id
- street
- city
- state
- postal_code
- country

### Vehicle

Represents a vehicle associated with a customer, policy, or claim.

Fields:

- vehicle_id
- customer_id
- policy_id
- make
- model
- year
- registration_id

### Document

Represents textual evidence used by the RAG system.

Fields:

- document_id
- claim_id
- document_type
- title
- source
- publication_date
- text
- checksum

### Image Evidence

Represents visual evidence associated with a claim.

Fields:

- image_id
- claim_id
- image_type
- file_path
- source
- checksum

## Relationships

Customer -> Policy
Customer -> Claim
Customer -> Address
Customer -> Vehicle

Policy -> Claim

Claim -> Provider
Claim -> Transaction
Claim -> Document
Claim -> Image

Provider -> Address

Transaction -> Customer
Transaction -> Claim
Transaction -> Provider

## Design Principles

1. External datasets are never used directly by downstream services.
2. Each source dataset gets its own ingestion/normalization adapter.
3. Canonical identifiers are stable within this platform.
4. Sensitive or secret credentials must never be stored in source code.
5. Raw datasets remain under data/raw and are excluded from Git.
6. Processed datasets remain under data/processed and are excluded from Git.
7. Graph, ML, RAG, and agent components consume canonical data rather than dataset-specific formats.
