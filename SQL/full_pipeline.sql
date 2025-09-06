-- 1) Core KPIs at (country, device, language)
CREATE OR REPLACE VIEW `nomadic-memento-469403-h1.ads_dataset.v_core_kpis` AS
SELECT
  ua_country,
  ua_device,
  placement_language,
  SUM(impressions)   AS impressions,
  SUM(engaged_views) AS engaged_views,
  SUM(conversions)   AS conversions,
  SUM(cost)          AS cost,
  SUM(revenue)       AS revenue,
  -- KPIs
  SAFE_DIVIDE(SUM(engaged_views), NULLIF(SUM(impressions),0))        AS engagement_rate,
  SAFE_DIVIDE(SUM(cost),          NULLIF(SUM(engaged_views),0))      AS cpe,
  SAFE_DIVIDE(SUM(cost),          NULLIF(SUM(conversions),0))        AS cpa,
  SAFE_DIVIDE(SUM(revenue) - SUM(cost), NULLIF(SUM(cost),0))         AS roi,
  SAFE_DIVIDE(SUM(revenue),       NULLIF(SUM(cost),0))               AS roas
FROM `nomadic-memento-469403-h1.ads_dataset.ads_raw`
GROUP BY 1,2,3;

-- 2) Heatmap helper (Country × Device) using engagement proxy
CREATE OR REPLACE VIEW `nomadic-memento-469403-h1.ads_dataset.v_heatmap_country_device_clean` AS
WITH base AS (
  SELECT
    ua_country,
    LOWER(ua_device) AS ua_device,
    LEAST(
      1.0,
      SAFE_DIVIDE(CAST(seconds_played AS FLOAT64), NULLIF(creative_duration,0))
    ) AS engagement_rate
  FROM `nomadic-memento-469403-h1.ads_dataset.ads_raw`
)
SELECT
  ua_country,
  ua_device,
  COUNT(*)               AS impressions,
  AVG(engagement_rate)   AS avg_engagement_rate
FROM base
GROUP BY 1,2;

-- 3) Device mix (for donut/pie)
CREATE OR REPLACE VIEW `nomadic-memento-469403-h1.ads_dataset.v_device_mix` AS
WITH counts AS (
  SELECT ua_device, SUM(impressions) AS impressions
  FROM `nomadic-memento-469403-h1.ads_dataset.ads_raw`
  GROUP BY ua_device
),
total AS (
  SELECT SUM(impressions) AS total_impressions FROM counts
)
SELECT
  c.ua_device,
  c.impressions,
  SAFE_DIVIDE(c.impressions, t.total_impressions) AS share_of_impressions
FROM counts c
CROSS JOIN total t
ORDER BY impressions DESC;

-- 4) Top languages by engagement proxy (and watch time)
CREATE OR REPLACE VIEW `nomadic-memento-469403-h1.ads_dataset.v_top_languages_engagement` AS
SELECT
  placement_language,
  COUNT(*) AS impressions,
  AVG(LEAST(1.0, SAFE_DIVIDE(CAST(seconds_played AS FLOAT64), NULLIF(creative_duration, 0)))) AS avg_engagement_rate,
  AVG(CAST(seconds_played AS FLOAT64)) AS avg_seconds_played
FROM `nomadic-memento-469403-h1.ads_dataset.ads_raw`
WHERE placement_language IS NOT NULL
GROUP BY placement_language
HAVING impressions > 5000
ORDER BY avg_engagement_rate DESC;

-- 5) Top countries (volume + watch time)
CREATE OR REPLACE VIEW `nomadic-memento-469403-h1.ads_dataset.v_top_countries` AS
SELECT
  ua_country,
  COUNT(*)             AS impressions,
  AVG(seconds_played)  AS avg_seconds_played
FROM `nomadic-memento-469403-h1.ads_dataset.ads_raw`
GROUP BY ua_country
HAVING impressions > 5000
ORDER BY impressions DESC;

-- 6) Top devices (engagement proxy)
CREATE OR REPLACE VIEW `nomadic-memento-469403-h1.ads_dataset.v_top_devices` AS
SELECT
  ua_device,
  COUNT(*) AS impressions,
  AVG(LEAST(1.0, SAFE_DIVIDE(CAST(seconds_played AS FLOAT64), NULLIF(creative_duration, 0)))) AS avg_engagement_rate
FROM `nomadic-memento-469403-h1.ads_dataset.ads_raw`
GROUP BY ua_device
HAVING impressions > 5000
ORDER BY avg_engagement_rate DESC;

-- 7) Languages by watch time (alt)
CREATE OR REPLACE VIEW `nomadic-memento-469403-h1.ads_dataset.v_top_languages` AS
SELECT
  placement_language,
  COUNT(*)            AS impressions,
  AVG(seconds_played) AS avg_seconds_played
FROM `nomadic-memento-469403-h1.ads_dataset.ads_raw`
GROUP BY placement_language
HAVING impressions > 5000
ORDER BY impressions DESC;

-- 8) Country × Device table (for clustered bar charts)
CREATE OR REPLACE VIEW `nomadic-memento-469403-h1.ads_dataset.v_country_device` AS
SELECT
  ua_country,
  ua_device,
  COUNT(*)             AS impressions,
  AVG(seconds_played)  AS avg_seconds_played
FROM `nomadic-memento-469403-h1.ads_dataset.ads_raw`
GROUP BY ua_country, ua_device
HAVING impressions > 2000
ORDER BY impressions DESC;

-- 9) “Top segments” for ROAS (guarding tiny/noisy rows)
CREATE OR REPLACE VIEW `nomadic-memento-469403-h1.ads_dataset.v_top_segments_roas_clean` AS
SELECT
  CONCAT(placement_language, ' | ', ua_device, ' | ', ua_country) AS segment,
  impressions, engaged_views, conversions, cost, revenue,
  engagement_rate, cpe, cpa, roi, roas
FROM `nomadic-memento-469403-h1.ads_dataset.v_core_kpis`
WHERE impressions >= 2000        -- scale guard
  AND cost >= 2.0                -- spend guard
  AND conversions >= 10;         -- conversion guard

-- 10) Bubble chart helper: Engagement vs ROAS by segment (with spend)
CREATE OR REPLACE VIEW `nomadic-memento-469403-h1.ads_dataset.v_bubble_engagement_roas_clean` AS
SELECT
  CONCAT(placement_language, ' | ', ua_device, ' | ', ua_country) AS segment,
  impressions,
  engagement_rate,
  roas,
  cost AS spend
FROM `nomadic-memento-469403-h1.ads_dataset.v_core_kpis`
WHERE impressions >= 2000 AND cost >= 2.0;


-- 11) Quick sanity checks -------------------------


-- 11a) Any tiny segments slipped through?
SELECT segment, impressions, cost, conversions, roas
FROM `nomadic-memento-469403-h1.ads_dataset.v_top_segments_roas_clean`
WHERE impressions < 2000 OR cost < 2.0 OR conversions < 10;

-- 11b) Top-10 ROAS (clean)
SELECT segment, impressions, cost, conversions, roas
FROM `nomadic-memento-469403-h1.ads_dataset.v_top_segments_roas_clean`
ORDER BY roas DESC
LIMIT 10;
