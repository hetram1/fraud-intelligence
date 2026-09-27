# Graph Investigation Guide

Document type: Synthetic graph-analysis guidance
Version: 1.0

## Purpose

The graph represents relationships among insurance claims, policies, incident locations, and incident attributes.

## Core Relationships

A claim may be connected to:

- a policy through HAS_CLAIM
- an incident location through OCCURRED_AT
- an incident type through HAS_INCIDENT_TYPE
- a collision type through HAS_COLLISION_TYPE
- an incident severity through HAS_SEVERITY
- an incident state through HAS_INCIDENT_STATE

## Shared Locations

Multiple claims associated with the same incident location can form a connected claim neighborhood.

Investigators should inspect the individual claims and dates associated with a shared location.

## Attribute Relationships

Claims can be grouped through common incident types, collision types, severity levels, and incident states.

These relationships provide contextual information for investigation and should not be interpreted as proof of fraudulent behavior.

## Community Analysis

Community detection can identify groups of closely connected claims.

A detected community is an analytical grouping. It is not itself a fraud label.

## Evidence Grounding

Any claim made by an investigation assistant should be grounded in retrieved graph facts or retrieved documents.

When evidence is unavailable, the system should explicitly state that the information is unavailable.
