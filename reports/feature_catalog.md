# Feature catalog

All interaction-derived features use accepted events strictly before the configured UTC cutoff. Entity metadata comes from accepted input CSVs. Tables are keyed as indicated and can be reloaded for batch prediction.

| Table | Key | Feature | Definition / missing policy |
|---|---|---|---|
| user_features | user_id | age, region | User metadata; age may be null, region defaults to Unknown. |
| user_features | user_id | activity_count | Number of training events; null for no history. |
| user_features | user_id | average_rating | Mean training rating; null for no history. |
| user_features | user_id | last_interaction | Most recent training timestamp UTC; null for no history. |
| user_features | user_id | days_since_last_interaction | Fractional days from last event to cutoff; null for no history. |
| user_features | user_id | preferred_category | Category with most training events; ties alphabetical; null for no history. |
| item_features | item_id | title, category | Accepted item metadata. |
| item_features | item_id | popularity | Training event count, zero for no history. |
| item_features | item_id | average_rating | Mean training rating; null for no history. |
| interaction_features | user_id, item_id | interaction_count | Number of training events for pair. |
| interaction_features | user_id, item_id | mean_rating | Mean training rating for pair. |
| interaction_features | user_id, item_id | last_interaction, days_since_last_interaction | Latest pair event and fractional days to cutoff. |
