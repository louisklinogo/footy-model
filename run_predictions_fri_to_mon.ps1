$ErrorActionPreference = "Stop"
Set-Location "C:\Developer\soccer\footy-model"

$days = 4
$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$outDir = "artifacts\reports\predictions"
New-Item -ItemType Directory -Force -Path $outDir | Out-Null

$csvOut = "$outDir\predictions_2026-02-27_to_2026-03-02_$ts.csv"
$mdOut  = "$outDir\predictions_2026-02-27_to_2026-03-02_$ts.md"
$signalsCsvOut = "$outDir\layer2_tracking_signals_2026-02-27_to_2026-03-02_$ts.csv"
$signalsMdOut  = "$outDir\layer2_tracking_signals_2026-02-27_to_2026-03-02_$ts.md"
$oddsCsvOut = "$outDir\market_odds_vs_model_2026-02-27_to_2026-03-02_$ts.csv"
$oddsMdOut  = "$outDir\market_odds_vs_model_2026-02-27_to_2026-03-02_$ts.md"
$oddsWideCsvOut = "$outDir\market_odds_vs_model_wide_2026-02-27_to_2026-03-02_$ts.csv"
$oddsWideMdOut  = "$outDir\market_odds_vs_model_wide_2026-02-27_to_2026-03-02_$ts.md"

# 1) Layer 2 residuals + rule layer
python src/modeling/layer2_situational/predict_situational_residual.py --days $days --enable-rule-layer --rule-overlap-mode override

# 2) Market probabilities
python src/modeling/evaluation/predict_market_outcomes_fixtures_first.py --days $days

# 3) Risk assessment
python src/modeling/evaluation/assess_prediction_risk.py --days $days

# 4) Export CSV
python src/modeling/export/export_market_outcomes_fixtures_first.py --days $days --out $csvOut

# 5) Markdown summary
@"
# Prediction Run Summary

- Window: 2026-02-27 to 2026-03-02
- Generated at: $(Get-Date -Format "yyyy-MM-dd HH:mm:ss")
- CSV: $csvOut

## Preview (first 20 rows)

"@ | Set-Content -Path $mdOut -Encoding UTF8

Get-Content $csvOut | Select-Object -First 21 | Add-Content -Path $mdOut -Encoding UTF8

# 6) Export Layer 2 tracking signals
python scripts/export_layer2_tracking_signals.py --days $days --out-csv $signalsCsvOut --out-md $signalsMdOut

# 7) Export model probability vs bookmaker odds/implied/edge
python scripts/export_market_model_odds_edges.py --days $days --out-csv $oddsCsvOut --out-md $oddsMdOut --out-csv-wide $oddsWideCsvOut --out-md-wide $oddsWideMdOut

Write-Host "Done."
Write-Host "CSV: $csvOut"
Write-Host "MD : $mdOut"
Write-Host "Signals CSV: $signalsCsvOut"
Write-Host "Signals MD : $signalsMdOut"
Write-Host "Odds CSV   : $oddsCsvOut"
Write-Host "Odds MD    : $oddsMdOut"
Write-Host "Odds Wide CSV: $oddsWideCsvOut"
Write-Host "Odds Wide MD : $oddsWideMdOut"
