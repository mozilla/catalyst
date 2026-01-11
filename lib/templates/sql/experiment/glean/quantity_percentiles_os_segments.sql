{% autoescape off %}
-- Query for quantity metrics in percentiles mode
-- Returns median, p75, p95 for the metric
WITH
{% if available_on_desktop == True %}
desktop_raw as (
    SELECT
        normalized_os as segment,
        mozfun.map.get_key(ping_info.experiments, "{{slug}}").branch as branch,
        {{histogram}} as value
    FROM `mozdata.firefox_desktop.metrics`
    WHERE
      DATE(submission_timestamp) >= DATE('{{startDate}}')
      AND DATE(submission_timestamp) <= DATE('{{endDate}}')
      AND normalized_channel = "{{channel}}"
      AND normalized_app_name = "Firefox"
      {% if single_os_filter %}AND normalized_os = "{{single_os_filter}}"{% endif %}
      AND {{histogram}} is not null
      AND mozfun.map.get_key(ping_info.experiments, "{{slug}}").branch is not null
),
{% else %}
desktop_raw as (
  SELECT "" as segment, "" as branch, 0 as value
  WHERE FALSE
),
{% endif %}
{% if available_on_android == True %}
android_raw as (
    SELECT
        normalized_os as segment,
        mozfun.map.get_key(ping_info.experiments, "{{slug}}").branch as branch,
        {{histogram}} as value
    FROM `mozdata.fenix.metrics`
    WHERE
      DATE(submission_timestamp) >= DATE('{{startDate}}')
      AND DATE(submission_timestamp) <= DATE('{{endDate}}')
      AND normalized_channel = "{{channel}}"
      {% if single_os_filter %}AND normalized_os = "{{single_os_filter}}"{% endif %}
      AND {{histogram}} is not null
      AND mozfun.map.get_key(ping_info.experiments, "{{slug}}").branch is not null
)
{% else %}
android_raw as (
  SELECT "" as segment, "" as branch, 0 as value
  WHERE FALSE
)
{% endif %}

SELECT
    segment,
    branch,
    APPROX_QUANTILES(value, 100)[OFFSET(50)] as median,
    APPROX_QUANTILES(value, 100)[OFFSET(75)] as p75,
    APPROX_QUANTILES(value, 100)[OFFSET(95)] as p95,
    SUM(value) as total_sum,
    COUNT(*) as sample_count
FROM (
    SELECT * FROM desktop_raw
    UNION ALL
    SELECT * FROM android_raw
)
GROUP BY segment, branch
ORDER BY segment, branch
{% endautoescape %}
