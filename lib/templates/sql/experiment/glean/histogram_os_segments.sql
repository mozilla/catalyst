{% autoescape off %}
with
{% if available_on_desktop == True %}
desktop_data as (
    SELECT
        normalized_os as segment,
        mozfun.map.get_key(ping_info.experiments, "{{slug}}").branch as branch,
        {% if distribution_type == "timing_distribution" %}
        CAST(key as INT64)/1000000 AS bucket,
        {% elif distribution_type == "labeled_counter" %}
        key AS bucket,
        {% else %}
        CAST(key as INT64) AS bucket,
        {% endif %}
        value as count
    FROM `mozdata.firefox_desktop.metrics` as d
      {% if distribution_type == "labeled_counter" %}
      CROSS JOIN UNNEST(d.{{histogram}})
      {% else %}
      CROSS JOIN UNNEST(d.{{histogram}}.values)
      {% endif %}
    WHERE
      DATE(submission_timestamp) >= DATE('{{startDate}}')
      AND DATE(submission_timestamp) <= DATE('{{endDate}}')
      AND normalized_channel = "{{channel}}"
      AND normalized_app_name = "Firefox"
      {% if single_os_filter %}AND normalized_os = "{{single_os_filter}}"{% endif %}
      AND d.{{histogram}} is not null
      AND ARRAY_LENGTH(ping_info.experiments) > 0
      AND mozfun.map.get_key(ping_info.experiments, "{{slug}}").branch is not null
),
{% else %}
desktop_data as (
  SELECT
    "" as segment,
    "" as branch,
    0 as bucket,
    0 as count
  FROM `mozdata.firefox_desktop.metrics` as d
  WHERE FALSE
),
{% endif %}
{% if available_on_android == True %}
android_data as (
    SELECT
        normalized_os as segment,
        mozfun.map.get_key(ping_info.experiments, "{{slug}}").branch as branch,
        {% if distribution_type == "timing_distribution" %}
        CAST(key as INT64)/1000000 AS bucket,
        {% elif distribution_type == "labeled_counter" %}
        key AS bucket,
        {% else %}
        CAST(key as INT64) AS bucket,
        {% endif %}
        value as count
    FROM `mozdata.fenix.metrics` as f
      {% if distribution_type == "labeled_counter" %}
      CROSS JOIN UNNEST(f.{{histogram}})
      {% else %}
      CROSS JOIN UNNEST(f.{{histogram}}.values)
      {% endif %}
    WHERE
      DATE(submission_timestamp) >= DATE('{{startDate}}')
      AND DATE(submission_timestamp) <= DATE('{{endDate}}')
      AND normalized_channel = "{{channel}}"
      {% if single_os_filter %}AND normalized_os = "{{single_os_filter}}"{% endif %}
      AND f.{{histogram}} is not null
      AND ARRAY_LENGTH(ping_info.experiments) > 0
      AND mozfun.map.get_key(ping_info.experiments, "{{slug}}").branch is not null
)
{% else %}
android_data as (
  SELECT
    "" as segment,
    "" as branch,
    0 as bucket,
    0 as count
  FROM `mozdata.fenix.metrics` as f
  WHERE FALSE
)
{% endif %}
{% if include_non_enrolled_branch == True %}
{% if available_on_desktop == True %}
,desktop_data_non_enrolled as (
    SELECT
        normalized_os as segment,
        "default" as branch,
        {% if distribution_type == "timing_distribution" %}
        CAST(key as INT64)/1000000 AS bucket,
        {% elif distribution_type == "labeled_counter" %}
        key AS bucket,
        {% else %}
        CAST(key as INT64) AS bucket,
        {% endif %}
        value as count
    FROM `mozdata.firefox_desktop.metrics` as d
      {% if distribution_type == "labeled_counter" %}
      CROSS JOIN UNNEST(d.{{histogram}})
      {% else %}
      CROSS JOIN UNNEST(d.{{histogram}}.values)
      {% endif %}
    WHERE
      DATE(submission_timestamp) >= DATE('{{startDate}}')
      AND DATE(submission_timestamp) <= DATE('{{endDate}}')
      AND normalized_channel = "{{channel}}"
      AND normalized_app_name = "Firefox"
      {% if single_os_filter %}AND normalized_os = "{{single_os_filter}}"{% endif %}
      AND d.{{histogram}} is not null
      AND ARRAY_LENGTH(ping_info.experiments) = 0
),
{% else %}
,desktop_data_non_enrolled as (
  SELECT 
    "" as segment,
    "" as branch,
    0 as bucket,
    0 as count
  FROM `mozdata.firefox_desktop.metrics` as d
  WHERE FALSE
),
{% endif %}
{% if available_on_android == True %}
android_data_non_enrolled as (
    SELECT
        normalized_os as segment,
        "default" as branch,
        {% if distribution_type == "timing_distribution" %}
        CAST(key as INT64)/1000000 AS bucket,
        {% elif distribution_type == "labeled_counter" %}
        key AS bucket,
        {% else %}
        CAST(key as INT64) AS bucket,
        {% endif %}
        value as count
    FROM `mozdata.fenix.metrics` as f
      {% if distribution_type == "labeled_counter" %}
      CROSS JOIN UNNEST(f.{{histogram}})
      {% else %}
      CROSS JOIN UNNEST(f.{{histogram}}.values)
      {% endif %}
    WHERE
      DATE(submission_timestamp) >= DATE('{{startDate}}')
      AND DATE(submission_timestamp) <= DATE('{{endDate}}')
      AND normalized_channel = "{{channel}}"
      {% if single_os_filter %}AND normalized_os = "{{single_os_filter}}"{% endif %}
      AND f.{{histogram}} is not null
      AND ARRAY_LENGTH(ping_info.experiments) = 0
)
{% else %}
android_data_non_enrolled as (
  SELECT
    "" as segment,
    "" as branch,
    0 as bucket,
    0 as count
  FROM `mozdata.fenix.metrics` as f
  WHERE FALSE
)
{% endif %}
{% endif %}

SELECT
    segment,
    branch,
    bucket,
    SUM(CAST(count AS NUMERIC)) as counts
FROM
    (
        SELECT * FROM desktop_data
        UNION ALL
        SELECT * FROM android_data
{% if include_non_enrolled_branch == True %}
        UNION ALL
        SELECT * FROM desktop_data_non_enrolled
        UNION ALL
        SELECT * FROM android_data_non_enrolled
{% endif %}
    ) s
GROUP BY
  segment, branch, bucket
ORDER BY
  segment, branch, bucket
{% endautoescape %}
