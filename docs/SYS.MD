
louis@script MINGW64 /c/Developer/soccer/footy-model (master)
$ python src/jobs/tick_due_fixtures_v1.py
COMMAND: node C:\Developer\soccer\footy-model\scrapers\premium_enricher_v4.js JP1 --ids-root C:\Developer\soccer\footy-model\data\v1\ids\tick\20260222T152027Z --out-root C:\Developer\soccer\footy-model\data\v1\premium
Starting Playwright Enrichment (v4) for 1 matches in JP1...
INFO  PlaywrightCrawler: Starting the crawler.
INFO  PlaywrightCrawler:Statistics: PlaywrightCrawler request statistics: {"requestAvgFailedDurationMillis":null,"requestAvgFinishedDurationMillis":null,"requestsFinishedPerMinute":0,"requestsFailedPerMinute":0,"requestTotalDurationMillis":0,"requestsTotal":0,"crawlerRuntimeMillis":60158,"retryHistogram":[]}
INFO  PlaywrightCrawler:AutoscaledPool: state {"currentConcurrency":1,"desiredConcurrency":3,"systemStatus":{"isSystemIdle":true,"memInfo":{"isOverloaded":false,"limitRatio":0.2,"actualRatio":0},"eventLoopInfo":{"isOverloaded":false,"limitRatio":0.6,"actualRatio":0},"cpuInfo":{"isOverloaded":false,"limitRatio":0.4,"actualRatio":0},"clientInfo":{"isOverloaded":false,"limitRatio":0.3,"actualRatio":0}}}
INFO  PlaywrightCrawler:Statistics: PlaywrightCrawler request statistics: {"requestAvgFailedDurationMillis":null,"requestAvgFinishedDurationMillis":null,"requestsFinishedPerMinute":0,"requestsFailedPerMinute":0,"requestTotalDurationMillis":0,"requestsTotal":0,"crawlerRuntimeMillis":120169,"retryHistogram":[]}
INFO  PlaywrightCrawler:AutoscaledPool: state {"currentConcurrency":1,"desiredConcurrency":3,"systemStatus":{"isSystemIdle":true,"memInfo":{"isOverloaded":false,"limitRatio":0.2,"actualRatio":0},"eventLoopInfo":{"isOverloaded":false,"limitRatio":0.6,"actualRatio":0},"cpuInfo":{"isOverloaded":false,"limitRatio":0.4,"actualRatio":0},"clientInfo":{"isOverloaded":false,"limitRatio":0.3,"actualRatio":0}}}
WARN  PlaywrightCrawler: Reclaiming failed request back to the list or queue. requestHandler timed out after 120 seconds. {"id":"WSY3gF3qqJ4F8J8","url":"https://www.flashscore.com/match/vH6L0RHk/","retryCount":1}
Failed vH6L0RHk: page.goto: Target page, context or browser has been closed
Call log:
  - navigating to "https://www.flashscore.com/match/football/kashima-antlers-4nNKlKo9/kashiwa-reysol-hEbtKvnM/odds/asian-handicap/full-time/?mid=vH6L0RHk", waiting until "networkidle"

INFO  PlaywrightCrawler:Statistics: PlaywrightCrawler request statistics: {"requestAvgFailedDurationMillis":null,"requestAvgFinishedDurationMillis":null,"requestsFinishedPerMinute":0,"requestsFailedPerMinute":0,"requestTotalDurationMillis":0,"requestsTotal":0,"crawlerRuntimeMillis":180174,"retryHistogram":[]}
INFO  PlaywrightCrawler:AutoscaledPool: state {"currentConcurrency":1,"desiredConcurrency":3,"systemStatus":{"isSystemIdle":true,"memInfo":{"isOverloaded":false,"limitRatio":0.2,"actualRatio":0},"eventLoopInfo":{"isOverloaded":false,"limitRatio":0.6,"actualRatio":0},"cpuInfo":{"isOverloaded":false,"limitRatio":0.4,"actualRatio":0},"clientInfo":{"isOverloaded":false,"limitRatio":0.3,"actualRatio":0}}}
Success: vH6L0RHk | Stats: 34 | OU: 17 | AH: 9
INFO  PlaywrightCrawler: All requests from the queue have been processed, the crawler will shut down.
INFO  PlaywrightCrawler: Final request statistics: {"requestsFinished":1,"requestsFailed":0,"retryHistogram":[null,1],"requestAvgFailedDurationMillis":null,"requestAvgFinishedDurationMillis":88252,"requestsFinishedPerMinute":0,"requestsFailedPerMinute":0,"requestTotalDurationMillis":88252,"requestsTotal":1,"crawlerRuntimeMillis":214673}
INFO  PlaywrightCrawler: Finished! Total 1 requests: 1 succeeded, 0 failed. {"terminal":true}
COMMAND: C:\Python313\python.exe C:\Developer\soccer\footy-model\src\ingest\ingest_premium_fixtures_v1.py --league JP1
{
  "league_code": "JP1",
  "premium_root": "data\\v1\\premium",
  "files": 1,
  "stats_upserts": 1,
  "odds_upserts": 1,
  "results_upserts": 1,
  "fixture_status_updates": 1,
  "skipped_missing_fixture": 0,
  "skipped_bad_json": 0
}
COMMAND: node C:\Developer\soccer\footy-model\scrapers\premium_enricher_v4.js MX1 --ids-root C:\Developer\soccer\footy-model\data\v1\ids\tick\20260222T152027Z --out-root C:\Developer\soccer\footy-model\data\v1\premium
Starting Playwright Enrichment (v4) for 2 matches in MX1...
INFO  PlaywrightCrawler: Starting the crawler.
INFO  PlaywrightCrawler:Statistics: PlaywrightCrawler request statistics: {"requestAvgFailedDurationMillis":null,"requestAvgFinishedDurationMillis":null,"requestsFinishedPerMinute":0,"requestsFailedPerMinute":0,"requestTotalDurationMillis":0,"requestsTotal":0,"crawlerRuntimeMillis":60149,"retryHistogram":[]}
INFO  PlaywrightCrawler:AutoscaledPool: state {"currentConcurrency":2,"desiredConcurrency":4,"systemStatus":{"isSystemIdle":true,"memInfo":{"isOverloaded":false,"limitRatio":0.2,"actualRatio":0},"eventLoopInfo":{"isOverloaded":false,"limitRatio":0.6,"actualRatio":0},"cpuInfo":{"isOverloaded":false,"limitRatio":0.4,"actualRatio":0},"clientInfo":{"isOverloaded":false,"limitRatio":0.3,"actualRatio":0}}}
Success: n3HLuEEk | Stats: 35 | OU: 20 | AH: 10
Success: YJFTwhq2 | Stats: 34 | OU: 17 | AH: 10
INFO  PlaywrightCrawler: All requests from the queue have been processed, the crawler will shut down.
INFO  PlaywrightCrawler: Final request statistics: {"requestsFinished":2,"requestsFailed":0,"retryHistogram":[2],"requestAvgFailedDurationMillis":null,"requestAvgFinishedDurationMillis":91321,"requestsFinishedPerMinute":1,"requestsFailedPerMinute":0,"requestTotalDurationMillis":182642,"requestsTotal":2,"crawlerRuntimeMillis":95211}
INFO  PlaywrightCrawler: Finished! Total 2 requests: 2 succeeded, 0 failed. {"terminal":true}
COMMAND: C:\Python313\python.exe C:\Developer\soccer\footy-model\src\ingest\ingest_premium_fixtures_v1.py --league MX1
{
  "league_code": "MX1",
  "premium_root": "data\\v1\\premium",
  "files": 2,
  "stats_upserts": 2,
  "odds_upserts": 2,
  "results_upserts": 2,
  "fixture_status_updates": 2,
  "skipped_missing_fixture": 0,
  "skipped_bad_json": 0
}
COMMAND: node C:\Developer\soccer\footy-model\scrapers\premium_enricher_v4.js T1 --ids-root C:\Developer\soccer\footy-model\data\v1\ids\tick\20260222T152027Z --out-root C:\Developer\soccer\footy-model\data\v1\premium
Starting Playwright Enrichment (v4) for 2 matches in T1...
INFO  PlaywrightCrawler: Starting the crawler.
INFO  PlaywrightCrawler:Statistics: PlaywrightCrawler request statistics: {"requestAvgFailedDurationMillis":null,"requestAvgFinishedDurationMillis":null,"requestsFinishedPerMinute":0,"requestsFailedPerMinute":0,"requestTotalDurationMillis":0,"requestsTotal":0,"crawlerRuntimeMillis":60143,"retryHistogram":[]}
INFO  PlaywrightCrawler:AutoscaledPool: state {"currentConcurrency":2,"desiredConcurrency":4,"systemStatus":{"isSystemIdle":true,"memInfo":{"isOverloaded":false,"limitRatio":0.2,"actualRatio":0},"eventLoopInfo":{"isOverloaded":false,"limitRatio":0.6,"actualRatio":0},"cpuInfo":{"isOverloaded":false,"limitRatio":0.4,"actualRatio":0},"clientInfo":{"isOverloaded":false,"limitRatio":0.3,"actualRatio":0}}}
Success: 4rjRVrAl | Stats: 33 | OU: 20 | AH: 9
Success: b5TaFNP7 | Stats: 33 | OU: 19 | AH: 9
INFO  PlaywrightCrawler: All requests from the queue have been processed, the crawler will shut down.
INFO  PlaywrightCrawler: Final request statistics: {"requestsFinished":2,"requestsFailed":0,"retryHistogram":[2],"requestAvgFailedDurationMillis":null,"requestAvgFinishedDurationMillis":98618,"requestsFinishedPerMinute":1,"requestsFailedPerMinute":0,"requestTotalDurationMillis":197235,"requestsTotal":2,"crawlerRuntimeMillis":118845}
INFO  PlaywrightCrawler: Finished! Total 2 requests: 2 succeeded, 0 failed. {"terminal":true}
COMMAND: C:\Python313\python.exe C:\Developer\soccer\footy-model\src\ingest\ingest_premium_fixtures_v1.py --league T1
{
  "league_code": "T1",
  "premium_root": "data\\v1\\premium",
  "files": 304,
  "stats_upserts": 304,
  "odds_upserts": 304,
  "results_upserts": 199,
  "fixture_status_updates": 2,
  "skipped_missing_fixture": 0,
  "skipped_bad_json": 0
}
COMMAND: node C:\Developer\soccer\footy-model\scrapers\premium_enricher_v4.js D2 --ids-root C:\Developer\soccer\footy-model\data\v1\ids\tick\20260222T152027Z --out-root C:\Developer\soccer\footy-model\data\v1\premium
Starting Playwright Enrichment (v4) for 3 matches in D2...
INFO  PlaywrightCrawler: Starting the crawler.
INFO  PlaywrightCrawler:Statistics: PlaywrightCrawler request statistics: {"requestAvgFailedDurationMillis":null,"requestAvgFinishedDurationMillis":null,"requestsFinishedPerMinute":0,"requestsFailedPerMinute":0,"requestTotalDurationMillis":0,"requestsTotal":0,"crawlerRuntimeMillis":60186,"retryHistogram":[]}
INFO  PlaywrightCrawler:AutoscaledPool: state {"currentConcurrency":2,"desiredConcurrency":4,"systemStatus":{"isSystemIdle":true,"memInfo":{"isOverloaded":false,"limitRatio":0.2,"actualRatio":0},"eventLoopInfo":{"isOverloaded":false,"limitRatio":0.6,"actualRatio":0},"cpuInfo":{"isOverloaded":false,"limitRatio":0.4,"actualRatio":0},"clientInfo":{"isOverloaded":false,"limitRatio":0.3,"actualRatio":0}}}
Failed nuamUgYP: page.goto: Timeout 60000ms exceeded.
Call log:
  - navigating to "https://www.flashscore.com/match/nuamUgYP/", waiting until "networkidle"

INFO  PlaywrightCrawler:Statistics: PlaywrightCrawler request statistics: {"requestAvgFailedDurationMillis":null,"requestAvgFinishedDurationMillis":72993,"requestsFinishedPerMinute":0,"requestsFailedPerMinute":0,"requestTotalDurationMillis":72993,"requestsTotal":1,"crawlerRuntimeMillis":120197,"retryHistogram":[1]}
INFO  PlaywrightCrawler:AutoscaledPool: state {"currentConcurrency":2,"desiredConcurrency":4,"systemStatus":{"isSystemIdle":true,"memInfo":{"isOverloaded":false,"limitRatio":0.2,"actualRatio":0},"eventLoopInfo":{"isOverloaded":false,"limitRatio":0.6,"actualRatio":0},"cpuInfo":{"isOverloaded":false,"limitRatio":0.4,"actualRatio":0},"clientInfo":{"isOverloaded":false,"limitRatio":0.3,"actualRatio":0}}}
Success: ELy18CB6 | Stats: 33 | OU: 18 | AH: 10
WARN  PlaywrightCrawler: Reclaiming failed request back to the list or queue. requestHandler timed out after 120 seconds. {"id":"UlXK9SMmqj3qj95","url":"https://www.flashscore.com/match/ELy18CB6/","retryCount":1}
Failed vwvSMBQa: page.goto: Timeout 60000ms exceeded.
Call log:
  - navigating to "https://www.flashscore.com/match/vwvSMBQa/", waiting until "networkidle"

INFO  PlaywrightCrawler:Statistics: PlaywrightCrawler request statistics: {"requestAvgFailedDurationMillis":null,"requestAvgFinishedDurationMillis":67776,"requestsFinishedPerMinute":1,"requestsFailedPerMinute":0,"requestTotalDurationMillis":135551,"requestsTotal":2,"crawlerRuntimeMillis":180211,"retryHistogram":[2]}
INFO  PlaywrightCrawler:AutoscaledPool: state {"currentConcurrency":1,"desiredConcurrency":4,"systemStatus":{"isSystemIdle":true,"memInfo":{"isOverloaded":false,"limitRatio":0.2,"actualRatio":0},"eventLoopInfo":{"isOverloaded":false,"limitRatio":0.6,"actualRatio":0},"cpuInfo":{"isOverloaded":false,"limitRatio":0.4,"actualRatio":0},"clientInfo":{"isOverloaded":false,"limitRatio":0.3,"actualRatio":0}}}
Success: ELy18CB6 | Stats: 33 | OU: 18 | AH: 10
INFO  PlaywrightCrawler: All requests from the queue have been processed, the crawler will shut down.
INFO  PlaywrightCrawler: Final request statistics: {"requestsFinished":3,"requestsFailed":0,"retryHistogram":[2,1],"requestAvgFailedDurationMillis":null,"requestAvgFinishedDurationMillis":69107,"requestsFinishedPerMinute":1,"requestsFailedPerMinute":0,"requestTotalDurationMillis":207322,"requestsTotal":3,"crawlerRuntimeMillis":206157}
INFO  PlaywrightCrawler: Finished! Total 3 requests: 3 succeeded, 0 failed. {"terminal":true}
COMMAND: C:\Python313\python.exe C:\Developer\soccer\footy-model\src\ingest\ingest_premium_fixtures_v1.py --league D2
{
  "league_code": "D2",
  "premium_root": "data\\v1\\premium",
  "files": 303,
  "stats_upserts": 303,
  "odds_upserts": 303,
  "results_upserts": 198,
  "fixture_status_updates": 1,
  "skipped_missing_fixture": 0,
  "skipped_bad_json": 0
}
COMMAND: node C:\Developer\soccer\footy-model\scrapers\premium_enricher_v4.js SP1 --ids-root C:\Developer\soccer\footy-model\data\v1\ids\tick\20260222T152027Z --out-root C:\Developer\soccer\footy-model\data\v1\premium
Starting Playwright Enrichment (v4) for 1 matches in SP1...
INFO  PlaywrightCrawler: Starting the crawler.
INFO  PlaywrightCrawler:Statistics: PlaywrightCrawler request statistics: {"requestAvgFailedDurationMillis":null,"requestAvgFinishedDurationMillis":null,"requestsFinishedPerMinute":0,"requestsFailedPerMinute":0,"requestTotalDurationMillis":0,"requestsTotal":0,"crawlerRuntimeMillis":60148,"retryHistogram":[]}
INFO  PlaywrightCrawler:AutoscaledPool: state {"currentConcurrency":1,"desiredConcurrency":3,"systemStatus":{"isSystemIdle":true,"memInfo":{"isOverloaded":false,"limitRatio":0.2,"actualRatio":0},"eventLoopInfo":{"isOverloaded":false,"limitRatio":0.6,"actualRatio":0},"cpuInfo":{"isOverloaded":false,"limitRatio":0.4,"actualRatio":0},"clientInfo":{"isOverloaded":false,"limitRatio":0.3,"actualRatio":0}}}
INFO  PlaywrightCrawler:Statistics: PlaywrightCrawler request statistics: {"requestAvgFailedDurationMillis":null,"requestAvgFinishedDurationMillis":null,"requestsFinishedPerMinute":0,"requestsFailedPerMinute":0,"requestTotalDurationMillis":0,"requestsTotal":0,"crawlerRuntimeMillis":120155,"retryHistogram":[]}
INFO  PlaywrightCrawler:AutoscaledPool: state {"currentConcurrency":1,"desiredConcurrency":3,"systemStatus":{"isSystemIdle":true,"memInfo":{"isOverloaded":false,"limitRatio":0.2,"actualRatio":0},"eventLoopInfo":{"isOverloaded":false,"limitRatio":0.6,"actualRatio":0},"cpuInfo":{"isOverloaded":false,"limitRatio":0.4,"actualRatio":0},"clientInfo":{"isOverloaded":false,"limitRatio":0.3,"actualRatio":0}}}
WARN  PlaywrightCrawler: Reclaiming failed request back to the list or queue. requestHandler timed out after 120 seconds. {"id":"ukUkfh6mtmRzkaL","url":"https://www.flashscore.com/match/WIPCQRL5/","retryCount":1}
Failed WIPCQRL5: page.goto: Target page, context or browser has been closed
Call log:
  - navigating to "https://www.flashscore.com/match/football/real-oviedo-SzYzw34K/real-sociedad-jNvak2f3/odds/asian-handicap/full-time/?mid=WIPCQRL5", waiting until "networkidle"

INFO  PlaywrightCrawler:Statistics: PlaywrightCrawler request statistics: {"requestAvgFailedDurationMillis":null,"requestAvgFinishedDurationMillis":null,"requestsFinishedPerMinute":0,"requestsFailedPerMinute":0,"requestTotalDurationMillis":0,"requestsTotal":0,"crawlerRuntimeMillis":180159,"retryHistogram":[]}
INFO  PlaywrightCrawler:AutoscaledPool: state {"currentConcurrency":1,"desiredConcurrency":3,"systemStatus":{"isSystemIdle":true,"memInfo":{"isOverloaded":false,"limitRatio":0.2,"actualRatio":0},"eventLoopInfo":{"isOverloaded":false,"limitRatio":0.6,"actualRatio":0},"cpuInfo":{"isOverloaded":false,"limitRatio":0.4,"actualRatio":0},"clientInfo":{"isOverloaded":false,"limitRatio":0.3,"actualRatio":0}}}
Failed WIPCQRL5: page.goto: Timeout 60000ms exceeded.
Call log:
  - navigating to "https://www.flashscore.com/match/WIPCQRL5/", waiting until "networkidle"

INFO  PlaywrightCrawler: All requests from the queue have been processed, the crawler will shut down.
INFO  PlaywrightCrawler: Final request statistics: {"requestsFinished":1,"requestsFailed":0,"retryHistogram":[null,1],"requestAvgFailedDurationMillis":null,"requestAvgFinishedDurationMillis":63672,"requestsFinishedPerMinute":0,"requestsFailedPerMinute":0,"requestTotalDurationMillis":63672,"requestsTotal":1,"crawlerRuntimeMillis":189171}
INFO  PlaywrightCrawler: Finished! Total 1 requests: 1 succeeded, 0 failed. {"terminal":true}
COMMAND: C:\Python313\python.exe C:\Developer\soccer\footy-model\src\ingest\ingest_premium_fixtures_v1.py --league SP1
{
  "league_code": "SP1",
  "premium_root": "data\\v1\\premium",
  "files": 217,
  "stats_upserts": 217,
  "odds_upserts": 217,
  "results_upserts": 107,
  "fixture_status_updates": 0,
  "skipped_missing_fixture": 0,
  "skipped_bad_json": 0
}
COMMAND: node C:\Developer\soccer\footy-model\scrapers\premium_enricher_v4.js SP2 --ids-root C:\Developer\soccer\footy-model\data\v1\ids\tick\20260222T152027Z --out-root C:\Developer\soccer\footy-model\data\v1\premium
Starting Playwright Enrichment (v4) for 1 matches in SP2...
INFO  PlaywrightCrawler: Starting the crawler.
INFO  PlaywrightCrawler:Statistics: PlaywrightCrawler request statistics: {"requestAvgFailedDurationMillis":null,"requestAvgFinishedDurationMillis":null,"requestsFinishedPerMinute":0,"requestsFailedPerMinute":0,"requestTotalDurationMillis":0,"requestsTotal":0,"crawlerRuntimeMillis":60145,"retryHistogram":[]}
INFO  PlaywrightCrawler:AutoscaledPool: state {"currentConcurrency":1,"desiredConcurrency":3,"systemStatus":{"isSystemIdle":true,"memInfo":{"isOverloaded":false,"limitRatio":0.2,"actualRatio":0},"eventLoopInfo":{"isOverloaded":false,"limitRatio":0.6,"actualRatio":0},"cpuInfo":{"isOverloaded":false,"limitRatio":0.4,"actualRatio":0},"clientInfo":{"isOverloaded":false,"limitRatio":0.3,"actualRatio":0}}}
Success: 4YWioJtT | Stats: 33 | OU: 19 | AH: 9
INFO  PlaywrightCrawler: All requests from the queue have been processed, the crawler will shut down.
INFO  PlaywrightCrawler: Final request statistics: {"requestsFinished":1,"requestsFailed":0,"retryHistogram":[1],"requestAvgFailedDurationMillis":null,"requestAvgFinishedDurationMillis":107533,"requestsFinishedPerMinute":1,"requestsFailedPerMinute":0,"requestTotalDurationMillis":107533,"requestsTotal":1,"crawlerRuntimeMillis":109774}
INFO  PlaywrightCrawler: Finished! Total 1 requests: 1 succeeded, 0 failed. {"terminal":true}
COMMAND: C:\Python313\python.exe C:\Developer\soccer\footy-model\src\ingest\ingest_premium_fixtures_v1.py --league SP2
{
  "league_code": "SP2",
  "premium_root": "data\\v1\\premium",
  "files": 462,
  "stats_upserts": 462,
  "odds_upserts": 462,
  "results_upserts": 286,
  "fixture_status_updates": 1,
  "skipped_missing_fixture": 0,
  "skipped_bad_json": 0
}
COMMAND: node C:\Developer\soccer\footy-model\scrapers\premium_enricher_v4.js F2 --ids-root C:\Developer\soccer\footy-model\data\v1\ids\tick\20260222T152027Z --out-root C:\Developer\soccer\footy-model\data\v1\premium
Starting Playwright Enrichment (v4) for 2 matches in F2...
INFO  PlaywrightCrawler: Starting the crawler.
INFO  PlaywrightCrawler:Statistics: PlaywrightCrawler request statistics: {"requestAvgFailedDurationMillis":null,"requestAvgFinishedDurationMillis":null,"requestsFinishedPerMinute":0,"requestsFailedPerMinute":0,"requestTotalDurationMillis":0,"requestsTotal":0,"crawlerRuntimeMillis":60135,"retryHistogram":[]}
INFO  PlaywrightCrawler:AutoscaledPool: state {"currentConcurrency":2,"desiredConcurrency":4,"systemStatus":{"isSystemIdle":true,"memInfo":{"isOverloaded":false,"limitRatio":0.2,"actualRatio":0},"eventLoopInfo":{"isOverloaded":false,"limitRatio":0.6,"actualRatio":0},"cpuInfo":{"isOverloaded":false,"limitRatio":0.4,"actualRatio":0},"clientInfo":{"isOverloaded":false,"limitRatio":0.3,"actualRatio":0}}}
Failed 2PIWpPv9: page.goto: Timeout 60000ms exceeded.
Call log:
  - navigating to "https://www.flashscore.com/match/2PIWpPv9/", waiting until "networkidle"

INFO  PlaywrightCrawler:Statistics: PlaywrightCrawler request statistics: {"requestAvgFailedDurationMillis":null,"requestAvgFinishedDurationMillis":65157,"requestsFinishedPerMinute":0,"requestsFailedPerMinute":0,"requestTotalDurationMillis":65157,"requestsTotal":1,"crawlerRuntimeMillis":120142,"retryHistogram":[1]}
INFO  PlaywrightCrawler:AutoscaledPool: state {"currentConcurrency":1,"desiredConcurrency":4,"systemStatus":{"isSystemIdle":true,"memInfo":{"isOverloaded":false,"limitRatio":0.2,"actualRatio":0},"eventLoopInfo":{"isOverloaded":false,"limitRatio":0.6,"actualRatio":0},"cpuInfo":{"isOverloaded":false,"limitRatio":0.4,"actualRatio":0},"clientInfo":{"isOverloaded":false,"limitRatio":0.3,"actualRatio":0}}}
WARN  PlaywrightCrawler: Reclaiming failed request back to the list or queue. requestHandler timed out after 120 seconds. {"id":"vg1nCiJjQhyHgli","url":"https://www.flashscore.com/match/G8fkYoe3/","retryCount":1}
Failed G8fkYoe3: page.waitForTimeout: Target page, context or browser has been closed
INFO  PlaywrightCrawler:Statistics: PlaywrightCrawler request statistics: {"requestAvgFailedDurationMillis":null,"requestAvgFinishedDurationMillis":65157,"requestsFinishedPerMinute":0,"requestsFailedPerMinute":0,"requestTotalDurationMillis":65157,"requestsTotal":1,"crawlerRuntimeMillis":180144,"retryHistogram":[1]}
INFO  PlaywrightCrawler:AutoscaledPool: state {"currentConcurrency":1,"desiredConcurrency":4,"systemStatus":{"isSystemIdle":true,"memInfo":{"isOverloaded":false,"limitRatio":0.2,"actualRatio":0},"eventLoopInfo":{"isOverloaded":false,"limitRatio":0.6,"actualRatio":0},"cpuInfo":{"isOverloaded":false,"limitRatio":0.4,"actualRatio":0},"clientInfo":{"isOverloaded":false,"limitRatio":0.3,"actualRatio":0}}}
Success: G8fkYoe3 | Stats: 33 | OU: 20 | AH: 10
INFO  PlaywrightCrawler: All requests from the queue have been processed, the crawler will shut down.
INFO  PlaywrightCrawler: Final request statistics: {"requestsFinished":2,"requestsFailed":0,"retryHistogram":[1,1],"requestAvgFailedDurationMillis":null,"requestAvgFinishedDurationMillis":86329,"requestsFinishedPerMinute":1,"requestsFailedPerMinute":0,"requestTotalDurationMillis":172657,"requestsTotal":2,"crawlerRuntimeMillis":233851}
INFO  PlaywrightCrawler: Finished! Total 2 requests: 2 succeeded, 0 failed. {"terminal":true}
COMMAND: C:\Python313\python.exe C:\Developer\soccer\footy-model\src\ingest\ingest_premium_fixtures_v1.py --league F2
{
  "league_code": "F2",
  "premium_root": "data\\v1\\premium",
  "files": 305,
  "stats_upserts": 305,
  "odds_upserts": 305,
  "results_upserts": 207,
  "fixture_status_updates": 1,
  "skipped_missing_fixture": 0,
  "skipped_bad_json": 0
}
COMMAND: node C:\Developer\soccer\footy-model\scrapers\premium_enricher_v4.js DK1 --ids-root C:\Developer\soccer\footy-model\data\v1\ids\tick\20260222T152027Z --out-root C:\Developer\soccer\footy-model\data\v1\premium
Starting Playwright Enrichment (v4) for 1 matches in DK1...
INFO  PlaywrightCrawler: Starting the crawler.
INFO  PlaywrightCrawler:Statistics: PlaywrightCrawler request statistics: {"requestAvgFailedDurationMillis":null,"requestAvgFinishedDurationMillis":null,"requestsFinishedPerMinute":0,"requestsFailedPerMinute":0,"requestTotalDurationMillis":0,"requestsTotal":0,"crawlerRuntimeMillis":60130,"retryHistogram":[]}
INFO  PlaywrightCrawler:AutoscaledPool: state {"currentConcurrency":1,"desiredConcurrency":3,"systemStatus":{"isSystemIdle":true,"memInfo":{"isOverloaded":false,"limitRatio":0.2,"actualRatio":0},"eventLoopInfo":{"isOverloaded":false,"limitRatio":0.6,"actualRatio":0},"cpuInfo":{"isOverloaded":false,"limitRatio":0.4,"actualRatio":0},"clientInfo":{"isOverloaded":false,"limitRatio":0.3,"actualRatio":0}}}
Success: MkmKSiim | Stats: 34 | OU: 19 | AH: 10
INFO  PlaywrightCrawler: All requests from the queue have been processed, the crawler will shut down.
INFO  PlaywrightCrawler: Final request statistics: {"requestsFinished":1,"requestsFailed":0,"retryHistogram":[1],"requestAvgFailedDurationMillis":null,"requestAvgFinishedDurationMillis":108368,"requestsFinishedPerMinute":1,"requestsFailedPerMinute":0,"requestTotalDurationMillis":108368,"requestsTotal":1,"crawlerRuntimeMillis":110530}
INFO  PlaywrightCrawler: Finished! Total 1 requests: 1 succeeded, 0 failed. {"terminal":true}
COMMAND: C:\Python313\python.exe C:\Developer\soccer\footy-model\src\ingest\ingest_premium_fixtures_v1.py --league DK1
{
  "league_code": "DK1",
  "premium_root": "data\\v1\\premium",
  "files": 132,
  "stats_upserts": 132,
  "odds_upserts": 132,
  "results_upserts": 122,
  "fixture_status_updates": 1,
  "skipped_missing_fixture": 0,
  "skipped_bad_json": 0
}
COMMAND: node C:\Developer\soccer\footy-model\scrapers\premium_enricher_v4.js E1 --ids-root C:\Developer\soccer\footy-model\data\v1\ids\tick\20260222T152027Z --out-root C:\Developer\soccer\footy-model\data\v1\premium
Starting Playwright Enrichment (v4) for 4 matches in E1...
INFO  PlaywrightCrawler: Starting the crawler.
INFO  PlaywrightCrawler:Statistics: PlaywrightCrawler request statistics: {"requestAvgFailedDurationMillis":null,"requestAvgFinishedDurationMillis":null,"requestsFinishedPerMinute":0,"requestsFailedPerMinute":0,"requestTotalDurationMillis":0,"requestsTotal":0,"crawlerRuntimeMillis":60141,"retryHistogram":[]}
INFO  PlaywrightCrawler:AutoscaledPool: state {"currentConcurrency":2,"desiredConcurrency":4,"systemStatus":{"isSystemIdle":true,"memInfo":{"isOverloaded":false,"limitRatio":0.2,"actualRatio":0},"eventLoopInfo":{"isOverloaded":false,"limitRatio":0.6,"actualRatio":0},"cpuInfo":{"isOverloaded":false,"limitRatio":0.4,"actualRatio":0},"clientInfo":{"isOverloaded":false,"limitRatio":0.3,"actualRatio":0}}}
Success: 6qMshZMj | Stats: 33 | OU: 20 | AH: 10
Success: hKYEm0tR | Stats: 33 | OU: 19 | AH: 9
INFO  PlaywrightCrawler:Statistics: PlaywrightCrawler request statistics: {"requestAvgFailedDurationMillis":null,"requestAvgFinishedDurationMillis":107035,"requestsFinishedPerMinute":1,"requestsFailedPerMinute":0,"requestTotalDurationMillis":214069,"requestsTotal":2,"crawlerRuntimeMillis":120147,"retryHistogram":[2]}
INFO  PlaywrightCrawler:AutoscaledPool: state {"currentConcurrency":2,"desiredConcurrency":5,"systemStatus":{"isSystemIdle":true,"memInfo":{"isOverloaded":false,"limitRatio":0.2,"actualRatio":0},"eventLoopInfo":{"isOverloaded":false,"limitRatio":0.6,"actualRatio":0},"cpuInfo":{"isOverloaded":false,"limitRatio":0.4,"actualRatio":0},"clientInfo":{"isOverloaded":false,"limitRatio":0.3,"actualRatio":0}}}
INFO  PlaywrightCrawler:Statistics: PlaywrightCrawler request statistics: {"requestAvgFailedDurationMillis":null,"requestAvgFinishedDurationMillis":107035,"requestsFinishedPerMinute":1,"requestsFailedPerMinute":0,"requestTotalDurationMillis":214069,"requestsTotal":2,"crawlerRuntimeMillis":180147,"retryHistogram":[2]}
INFO  PlaywrightCrawler:AutoscaledPool: state {"currentConcurrency":2,"desiredConcurrency":5,"systemStatus":{"isSystemIdle":true,"memInfo":{"isOverloaded":false,"limitRatio":0.2,"actualRatio":0},"eventLoopInfo":{"isOverloaded":false,"limitRatio":0.6,"actualRatio":0},"cpuInfo":{"isOverloaded":false,"limitRatio":0.4,"actualRatio":0},"clientInfo":{"isOverloaded":false,"limitRatio":0.3,"actualRatio":0}}}
Success: n5rLl0Gb | Stats: 33 | OU: 20 | AH: 10
WARN  PlaywrightCrawler: Reclaiming failed request back to the list or queue. requestHandler timed out after 120 seconds. {"id":"XFOJAm80rnvsiNE","url":"https://www.flashscore.com/match/vZy3aHUG/","retryCount":1}
Failed vZy3aHUG: page.goto: Target page, context or browser has been closed
Call log:
  - navigating to "https://www.flashscore.com/match/football/leicester-KrrdAMyI/stoke-city-hSUajdSS/odds/asian-handicap/full-time/?mid=vZy3aHUG", waiting until "networkidle"

INFO  PlaywrightCrawler:Statistics: PlaywrightCrawler request statistics: {"requestAvgFailedDurationMillis":null,"requestAvgFinishedDurationMillis":98251,"requestsFinishedPerMinute":1,"requestsFailedPerMinute":0,"requestTotalDurationMillis":294753,"requestsTotal":3,"crawlerRuntimeMillis":240157,"retryHistogram":[3]}
INFO  PlaywrightCrawler:AutoscaledPool: state {"currentConcurrency":1,"desiredConcurrency":5,"systemStatus":{"isSystemIdle":true,"memInfo":{"isOverloaded":false,"limitRatio":0.2,"actualRatio":0},"eventLoopInfo":{"isOverloaded":false,"limitRatio":0.6,"actualRatio":0},"cpuInfo":{"isOverloaded":false,"limitRatio":0.4,"actualRatio":0},"clientInfo":{"isOverloaded":false,"limitRatio":0.3,"actualRatio":0}}}
Failed vZy3aHUG: page.goto: Timeout 60000ms exceeded.
Call log:
  - navigating to "https://www.flashscore.com/match/vZy3aHUG/", waiting until "networkidle"

INFO  PlaywrightCrawler: All requests from the queue have been processed, the crawler will shut down.
INFO  PlaywrightCrawler: Final request statistics: {"requestsFinished":4,"requestsFailed":0,"retryHistogram":[3,1],"requestAvgFailedDurationMillis":null,"requestAvgFinishedDurationMillis":89339,"requestsFinishedPerMinute":1,"requestsFailedPerMinute":0,"requestTotalDurationMillis":357354,"requestsTotal":4,"crawlerRuntimeMillis":289761}
INFO  PlaywrightCrawler: Finished! Total 4 requests: 4 succeeded, 0 failed. {"terminal":true}
COMMAND: C:\Python313\python.exe C:\Developer\soccer\footy-model\src\ingest\ingest_premium_fixtures_v1.py --league E1
{
  "league_code": "E1",
  "premium_root": "data\\v1\\premium",
  "files": 549,
  "stats_upserts": 549,
  "odds_upserts": 549,
  "results_upserts": 384,
  "fixture_status_updates": 3,
  "skipped_missing_fixture": 0,
  "skipped_bad_json": 0
}
COMMAND: node C:\Developer\soccer\footy-model\scrapers\premium_enricher_v4.js E2 --ids-root C:\Developer\soccer\footy-model\data\v1\ids\tick\20260222T152027Z --out-root C:\Developer\soccer\footy-model\data\v1\premium
Starting Playwright Enrichment (v4) for 3 matches in E2...
INFO  PlaywrightCrawler: Starting the crawler.
INFO  PlaywrightCrawler:Statistics: PlaywrightCrawler request statistics: {"requestAvgFailedDurationMillis":null,"requestAvgFinishedDurationMillis":null,"requestsFinishedPerMinute":0,"requestsFailedPerMinute":0,"requestTotalDurationMillis":0,"requestsTotal":0,"crawlerRuntimeMillis":60175,"retryHistogram":[]}
INFO  PlaywrightCrawler:AutoscaledPool: state {"currentConcurrency":2,"desiredConcurrency":4,"systemStatus":{"isSystemIdle":true,"memInfo":{"isOverloaded":false,"limitRatio":0.2,"actualRatio":0},"eventLoopInfo":{"isOverloaded":false,"limitRatio":0.6,"actualRatio":0},"cpuInfo":{"isOverloaded":false,"limitRatio":0.4,"actualRatio":0},"clientInfo":{"isOverloaded":false,"limitRatio":0.3,"actualRatio":0}}}
Success: dESgeNN0 | Stats: 33 | OU: 21 | AH: 10
INFO  PlaywrightCrawler:Statistics: PlaywrightCrawler request statistics: {"requestAvgFailedDurationMillis":null,"requestAvgFinishedDurationMillis":99193,"requestsFinishedPerMinute":0,"requestsFailedPerMinute":0,"requestTotalDurationMillis":99193,"requestsTotal":1,"crawlerRuntimeMillis":120180,"retryHistogram":[1]}
INFO  PlaywrightCrawler:AutoscaledPool: state {"currentConcurrency":2,"desiredConcurrency":4,"systemStatus":{"isSystemIdle":true,"memInfo":{"isOverloaded":false,"limitRatio":0.2,"actualRatio":0},"eventLoopInfo":{"isOverloaded":false,"limitRatio":0.6,"actualRatio":0},"cpuInfo":{"isOverloaded":false,"limitRatio":0.4,"actualRatio":0},"clientInfo":{"isOverloaded":false,"limitRatio":0.3,"actualRatio":0}}}
Failed fwfTBCek: page.goto: Timeout 60000ms exceeded.
Call log:
  - navigating to "https://www.flashscore.com/match/football/cardiff-hO8wh6aP/plymouth-WpGM6ELj/odds/asian-handicap/full-time/?mid=fwfTBCek", waiting until "networkidle"

INFO  PlaywrightCrawler:Statistics: PlaywrightCrawler request statistics: {"requestAvgFailedDurationMillis":null,"requestAvgFinishedDurationMillis":111172,"requestsFinishedPerMinute":1,"requestsFailedPerMinute":0,"requestTotalDurationMillis":222343,"requestsTotal":2,"crawlerRuntimeMillis":180185,"retryHistogram":[2]}
INFO  PlaywrightCrawler:AutoscaledPool: state {"currentConcurrency":1,"desiredConcurrency":4,"systemStatus":{"isSystemIdle":true,"memInfo":{"isOverloaded":false,"limitRatio":0.2,"actualRatio":0},"eventLoopInfo":{"isOverloaded":false,"limitRatio":0.6,"actualRatio":0},"cpuInfo":{"isOverloaded":false,"limitRatio":0.4,"actualRatio":0},"clientInfo":{"isOverloaded":false,"limitRatio":0.3,"actualRatio":0}}}
Success: r9XD0Iat | Stats: 33 | OU: 18 | AH: 9
INFO  PlaywrightCrawler: All requests from the queue have been processed, the crawler will shut down.
INFO  PlaywrightCrawler: Final request statistics: {"requestsFinished":3,"requestsFailed":0,"retryHistogram":[3],"requestAvgFailedDurationMillis":null,"requestAvgFinishedDurationMillis":107909,"requestsFinishedPerMinute":1,"requestsFailedPerMinute":0,"requestTotalDurationMillis":323726,"requestsTotal":3,"crawlerRuntimeMillis":202854}
INFO  PlaywrightCrawler: Finished! Total 3 requests: 3 succeeded, 0 failed. {"terminal":true}
COMMAND: C:\Python313\python.exe C:\Developer\soccer\footy-model\src\ingest\ingest_premium_fixtures_v1.py --league E2
{
  "league_code": "E2",
  "premium_root": "data\\v1\\premium",
  "files": 545,
  "stats_upserts": 545,
  "odds_upserts": 545,
  "results_upserts": 373,
  "fixture_status_updates": 2,
  "skipped_missing_fixture": 0,
  "skipped_bad_json": 0
}
COMMAND: node C:\Developer\soccer\footy-model\scrapers\premium_enricher_v4.js E3 --ids-root C:\Developer\soccer\footy-model\data\v1\ids\tick\20260222T152027Z --out-root C:\Developer\soccer\footy-model\data\v1\premium
Starting Playwright Enrichment (v4) for 2 matches in E3...
INFO  PlaywrightCrawler: Starting the crawler.
INFO  PlaywrightCrawler:Statistics: PlaywrightCrawler request statistics: {"requestAvgFailedDurationMillis":null,"requestAvgFinishedDurationMillis":null,"requestsFinishedPerMinute":0,"requestsFailedPerMinute":0,"requestTotalDurationMillis":0,"requestsTotal":0,"crawlerRuntimeMillis":60143,"retryHistogram":[]}
INFO  PlaywrightCrawler:AutoscaledPool: state {"currentConcurrency":2,"desiredConcurrency":4,"systemStatus":{"isSystemIdle":true,"memInfo":{"isOverloaded":false,"limitRatio":0.2,"actualRatio":0},"eventLoopInfo":{"isOverloaded":false,"limitRatio":0.6,"actualRatio":0},"cpuInfo":{"isOverloaded":false,"limitRatio":0.4,"actualRatio":0},"clientInfo":{"isOverloaded":false,"limitRatio":0.3,"actualRatio":0}}}
Success: OdSsZvGk | Stats: 34 | OU: 18 | AH: 9
INFO  PlaywrightCrawler:Statistics: PlaywrightCrawler request statistics: {"requestAvgFailedDurationMillis":null,"requestAvgFinishedDurationMillis":80633,"requestsFinishedPerMinute":0,"requestsFailedPerMinute":0,"requestTotalDurationMillis":80633,"requestsTotal":1,"crawlerRuntimeMillis":120150,"retryHistogram":[1]}
INFO  PlaywrightCrawler:AutoscaledPool: state {"currentConcurrency":1,"desiredConcurrency":4,"systemStatus":{"isSystemIdle":true,"memInfo":{"isOverloaded":false,"limitRatio":0.2,"actualRatio":0},"eventLoopInfo":{"isOverloaded":false,"limitRatio":0.6,"actualRatio":0},"cpuInfo":{"isOverloaded":false,"limitRatio":0.4,"actualRatio":0},"clientInfo":{"isOverloaded":false,"limitRatio":0.3,"actualRatio":0}}}
WARN  PlaywrightCrawler: Reclaiming failed request back to the list or queue. requestHandler timed out after 120 seconds. {"id":"QfoCaBvaRKNahwn","url":"https://www.flashscore.com/match/4U2GmvVr/","retryCount":1}
Failed 4U2GmvVr: page.goto: Target page, context or browser has been closed
Call log:
  - navigating to "https://www.flashscore.com/match/football/crewe-YacqHHdS/swindon-xSloasB8/odds/asian-handicap/full-time/?mid=4U2GmvVr", waiting until "networkidle"

INFO  PlaywrightCrawler:Statistics: PlaywrightCrawler request statistics: {"requestAvgFailedDurationMillis":null,"requestAvgFinishedDurationMillis":80633,"requestsFinishedPerMinute":0,"requestsFailedPerMinute":0,"requestTotalDurationMillis":80633,"requestsTotal":1,"crawlerRuntimeMillis":180152,"retryHistogram":[1]}
INFO  PlaywrightCrawler:AutoscaledPool: state {"currentConcurrency":1,"desiredConcurrency":4,"systemStatus":{"isSystemIdle":true,"memInfo":{"isOverloaded":false,"limitRatio":0.2,"actualRatio":0},"eventLoopInfo":{"isOverloaded":false,"limitRatio":0.6,"actualRatio":0},"cpuInfo":{"isOverloaded":false,"limitRatio":0.4,"actualRatio":0},"clientInfo":{"isOverloaded":false,"limitRatio":0.3,"actualRatio":0}}}
INFO  PlaywrightCrawler:Statistics: PlaywrightCrawler request statistics: {"requestAvgFailedDurationMillis":null,"requestAvgFinishedDurationMillis":80633,"requestsFinishedPerMinute":0,"requestsFailedPerMinute":0,"requestTotalDurationMillis":80633,"requestsTotal":1,"crawlerRuntimeMillis":240162,"retryHistogram":[1]}
INFO  PlaywrightCrawler:AutoscaledPool: state {"currentConcurrency":1,"desiredConcurrency":4,"systemStatus":{"isSystemIdle":true,"memInfo":{"isOverloaded":false,"limitRatio":0.2,"actualRatio":0},"eventLoopInfo":{"isOverloaded":false,"limitRatio":0.6,"actualRatio":0},"cpuInfo":{"isOverloaded":false,"limitRatio":0.4,"actualRatio":0},"clientInfo":{"isOverloaded":false,"limitRatio":0.3,"actualRatio":0}}}
Success: 4U2GmvVr | Stats: 33 | OU: 19 | AH: 9
INFO  PlaywrightCrawler: All requests from the queue have been processed, the crawler will shut down.
INFO  PlaywrightCrawler: Final request statistics: {"requestsFinished":2,"requestsFailed":0,"retryHistogram":[1,1],"requestAvgFailedDurationMillis":null,"requestAvgFinishedDurationMillis":99329,"requestsFinishedPerMinute":0,"requestsFailedPerMinute":0,"requestTotalDurationMillis":198657,"requestsTotal":2,"crawlerRuntimeMillis":243922}
INFO  PlaywrightCrawler: Finished! Total 2 requests: 2 succeeded, 0 failed. {"terminal":true}
COMMAND: C:\Python313\python.exe C:\Developer\soccer\footy-model\src\ingest\ingest_premium_fixtures_v1.py --league E3
{
  "league_code": "E3",
  "premium_root": "data\\v1\\premium",
  "files": 550,
  "stats_upserts": 550,
  "odds_upserts": 550,
  "results_upserts": 382,
  "fixture_status_updates": 2,
  "skipped_missing_fixture": 0,
  "skipped_bad_json": 0
}
COMMAND: node C:\Developer\soccer\footy-model\scrapers\premium_enricher_v4.js I1 --ids-root C:\Developer\soccer\footy-model\data\v1\ids\tick\20260222T152027Z --out-root C:\Developer\soccer\footy-model\data\v1\premium
Starting Playwright Enrichment (v4) for 1 matches in I1...
INFO  PlaywrightCrawler: Starting the crawler.
INFO  PlaywrightCrawler:Statistics: PlaywrightCrawler request statistics: {"requestAvgFailedDurationMillis":null,"requestAvgFinishedDurationMillis":null,"requestsFinishedPerMinute":0,"requestsFailedPerMinute":0,"requestTotalDurationMillis":0,"requestsTotal":0,"crawlerRuntimeMillis":60143,"retryHistogram":[]}
INFO  PlaywrightCrawler:AutoscaledPool: state {"currentConcurrency":1,"desiredConcurrency":3,"systemStatus":{"isSystemIdle":true,"memInfo":{"isOverloaded":false,"limitRatio":0.2,"actualRatio":0},"eventLoopInfo":{"isOverloaded":false,"limitRatio":0.6,"actualRatio":0},"cpuInfo":{"isOverloaded":false,"limitRatio":0.4,"actualRatio":0},"clientInfo":{"isOverloaded":false,"limitRatio":0.3,"actualRatio":0}}}
Success: Q1j59KFM | Stats: 33 | OU: 17 | AH: 9
INFO  PlaywrightCrawler: All requests from the queue have been processed, the crawler will shut down.
INFO  PlaywrightCrawler: Final request statistics: {"requestsFinished":1,"requestsFailed":0,"retryHistogram":[1],"requestAvgFailedDurationMillis":null,"requestAvgFinishedDurationMillis":115744,"requestsFinishedPerMinute":1,"requestsFailedPerMinute":0,"requestTotalDurationMillis":115744,"requestsTotal":1,"crawlerRuntimeMillis":118012}
INFO  PlaywrightCrawler: Finished! Total 1 requests: 1 succeeded, 0 failed. {"terminal":true}
COMMAND: C:\Python313\python.exe C:\Developer\soccer\footy-model\src\ingest\ingest_premium_fixtures_v1.py --league I1
{
  "league_code": "I1",
  "premium_root": "data\\v1\\premium",
  "files": 380,
  "stats_upserts": 380,
  "odds_upserts": 380,
  "results_upserts": 251,
  "fixture_status_updates": 1,
  "skipped_missing_fixture": 0,
  "skipped_bad_json": 0
}
COMMAND: node C:\Developer\soccer\footy-model\scrapers\premium_enricher_v4.js I2 --ids-root C:\Developer\soccer\footy-model\data\v1\ids\tick\20260222T152027Z --out-root C:\Developer\soccer\footy-model\data\v1\premium
Starting Playwright Enrichment (v4) for 2 matches in I2...
INFO  PlaywrightCrawler: Starting the crawler.
INFO  PlaywrightCrawler:Statistics: PlaywrightCrawler request statistics: {"requestAvgFailedDurationMillis":null,"requestAvgFinishedDurationMillis":null,"requestsFinishedPerMinute":0,"requestsFailedPerMinute":0,"requestTotalDurationMillis":0,"requestsTotal":0,"crawlerRuntimeMillis":60153,"retryHistogram":[]}
INFO  PlaywrightCrawler:AutoscaledPool: state {"currentConcurrency":2,"desiredConcurrency":4,"systemStatus":{"isSystemIdle":true,"memInfo":{"isOverloaded":false,"limitRatio":0.2,"actualRatio":0},"eventLoopInfo":{"isOverloaded":false,"limitRatio":0.6,"actualRatio":0},"cpuInfo":{"isOverloaded":false,"limitRatio":0.4,"actualRatio":0},"clientInfo":{"isOverloaded":false,"limitRatio":0.3,"actualRatio":0}}}
Success: 2TxVB3CM | Stats: 33 | OU: 20 | AH: 9
Success: dG7UDPsA | Stats: 33 | OU: 19 | AH: 9
INFO  PlaywrightCrawler: All requests from the queue have been processed, the crawler will shut down.
INFO  PlaywrightCrawler: Final request statistics: {"requestsFinished":2,"requestsFailed":0,"retryHistogram":[2],"requestAvgFailedDurationMillis":null,"requestAvgFinishedDurationMillis":78871,"requestsFinishedPerMinute":1,"requestsFailedPerMinute":0,"requestTotalDurationMillis":157742,"requestsTotal":2,"crawlerRuntimeMillis":96727}
INFO  PlaywrightCrawler: Finished! Total 2 requests: 2 succeeded, 0 failed. {"terminal":true}
COMMAND: C:\Python313\python.exe C:\Developer\soccer\footy-model\src\ingest\ingest_premium_fixtures_v1.py --league I2
{
  "league_code": "I2",
  "premium_root": "data\\v1\\premium",
  "files": 379,
  "stats_upserts": 379,
  "odds_upserts": 379,
  "results_upserts": 251,
  "fixture_status_updates": 2,
  "skipped_missing_fixture": 0,
  "skipped_bad_json": 0
}
COMMAND: C:\Python313\python.exe C:\Developer\soccer\footy-model\scrapers\build_team_premium_snapshots_v1.py --league E0
C:\Python313\python.exe: can't open file 'C:\\Developer\\soccer\\footy-model\\scrapers\\build_team_premium_snapshots_v1.py': [Errno 2] No such file or directory
Tick failed: Command '['C:\\Python313\\python.exe', 'C:\\Developer\\soccer\\footy-model\\scrapers\\build_team_premium_snapshots_v1.py', '--league', 'E0']' returned non-zero exit status 2.

louis@script MINGW64 /c/Developer/soccer/footy-model (master)
$