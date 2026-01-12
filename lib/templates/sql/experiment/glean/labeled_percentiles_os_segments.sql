{% autoescape off %}
-- Query for labeled_counter metrics in histogram mode
-- Returns median, p75, p95 for each label
-- Optimized: filter to enrolled clients first, then unnest
WITH
{% if available_on_desktop == True %}
desktop_enrolled as (
    SELECT
        normalized_os as segment,
        mozfun.map.get_key(ping_info.experiments, "{{slug}}").branch as branch,
        {{histogram}} as counter_data
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
desktop_raw as (
    SELECT
        segment,
        branch,
        key AS label,
        value
    FROM desktop_enrolled
    CROSS JOIN UNNEST(counter_data)
    WHERE value > 0
),
{% if include_non_enrolled_branch == True %}
desktop_raw_non_enrolled as (
    SELECT
        normalized_os as segment,
        "default" as branch,
        key AS label,
        value
    FROM `mozdata.firefox_desktop.metrics`
    CROSS JOIN UNNEST({{histogram}}) as counter_data
    WHERE
      DATE(submission_timestamp) >= DATE('{{startDate}}')
      AND DATE(submission_timestamp) <= DATE('{{endDate}}')
      AND normalized_channel = "{{channel}}"
      AND normalized_app_name = "Firefox"
      {% if single_os_filter %}AND normalized_os = "{{single_os_filter}}"{% endif %}
      AND {{histogram}} is not null
      AND mozfun.map.get_key(ping_info.experiments, "{{slug}}").branch is null
      AND value > 0
),
{% endif %}
{% else %}
desktop_raw as (
  SELECT "" as segment, "" as branch, "" as label, 0 as value
  WHERE FALSE
),
{% if include_non_enrolled_branch == True %}
desktop_raw_non_enrolled as (
  SELECT "" as segment, "" as branch, "" as label, 0 as value
  WHERE FALSE
),
{% endif %}
{% endif %}
{% if available_on_android == True %}
android_enrolled as (
    SELECT
        normalized_os as segment,
        mozfun.map.get_key(ping_info.experiments, "{{slug}}").branch as branch,
        {{histogram}} as counter_data
    FROM `mozdata.fenix.metrics`
    WHERE
      DATE(submission_timestamp) >= DATE('{{startDate}}')
      AND DATE(submission_timestamp) <= DATE('{{endDate}}')
      AND normalized_channel = "{{channel}}"
      {% if single_os_filter %}AND normalized_os = "{{single_os_filter}}"{% endif %}
      AND {{histogram}} is not null
      AND mozfun.map.get_key(ping_info.experiments, "{{slug}}").branch is not null
),
android_raw as (
    SELECT
        segment,
        branch,
        key AS label,
        value
    FROM android_enrolled
    CROSS JOIN UNNEST(counter_data)
    WHERE value > 0
),
{% if include_non_enrolled_branch == True %}
android_raw_non_enrolled as (
    SELECT
        normalized_os as segment,
        "default" as branch,
        key AS label,
        value
    FROM `mozdata.fenix.metrics`
    CROSS JOIN UNNEST({{histogram}}) as counter_data
    WHERE
      DATE(submission_timestamp) >= DATE('{{startDate}}')
      AND DATE(submission_timestamp) <= DATE('{{endDate}}')
      AND normalized_channel = "{{channel}}"
      {% if single_os_filter %}AND normalized_os = "{{single_os_filter}}"{% endif %}
      AND {{histogram}} is not null
      AND mozfun.map.get_key(ping_info.experiments, "{{slug}}").branch is null
      AND value > 0
)
{% endif %}
{% else %}
android_raw as (
  SELECT "" as segment, "" as branch, "" as label, 0 as value
  WHERE FALSE
),
{% if include_non_enrolled_branch == True %}
android_raw_non_enrolled as (
  SELECT "" as segment, "" as branch, "" as label, 0 as value
  WHERE FALSE
)
{% endif %}
{% endif %}

SELECT
    segment,
    branch,
    label,
    APPROX_QUANTILES(value, 100)[OFFSET(50)] as median,
    APPROX_QUANTILES(value, 100)[OFFSET(75)] as p75,
    APPROX_QUANTILES(value, 100)[OFFSET(95)] as p95,
    COUNT(*) as sample_count
FROM (
    SELECT * FROM desktop_raw
    UNION ALL
    SELECT * FROM android_raw
{% if include_non_enrolled_branch == True %}
    UNION ALL
    SELECT * FROM desktop_raw_non_enrolled
    UNION ALL
    SELECT * FROM android_raw_non_enrolled
{% endif %}
)
GROUP BY segment, branch, label
ORDER BY segment, branch, label
{% endautoescape %}
