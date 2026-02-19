#!/bin/bash
# Mass Enrichment Script for Major Leagues
LEAGUES=("E0" "SP1" "D1" "I1" "F1" "N1")

for LEAGUE in "${LEAGUES[@]}"
do
    echo "--------------------------------------------------"
    echo "🚀 ENRICHING LEAGUE: $LEAGUE"
    echo "--------------------------------------------------"
    node footy-model-v2/scrapers/premium_enricher_v2.js $LEAGUE
done

echo "🏁 Major League Enrichment Complete!"
