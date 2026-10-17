{% autoescape off %}
WITH
{% if available_on_desktop == True %}
desktop_base AS (
    SELECT
        client_info.client_id as client_id,
        normalized_os as segment,
        mozfun.map.get_key(ping_info.experiments, "{{slug}}").branch as branch,
        {% for h in histograms %}
        d.{{h.name}} as {{h.alias}}{% if not forloop.last %},{% endif %}
        {% endfor %}
    FROM `mozdata.firefox_desktop.metrics` as d
    WHERE
      DATE(submission_timestamp) >= DATE('{{startDate}}')
      AND DATE(submission_timestamp) <= DATE('{{endDate}}')
      AND normalized_channel = "{{channel}}"
      AND normalized_app_name = "Firefox"
      {% if single_os_filter %}AND normalized_os = "{{single_os_filter}}"{% endif %}
      AND ARRAY_LENGTH(ping_info.experiments) > 0
      AND mozfun.map.get_key(ping_info.experiments, "{{slug}}").branch is not null
      AND (
        {% for h in histograms %}
        d.{{h.name}} is not null{% if not forloop.last %} OR {% endif %}
        {% endfor %}
      )
),
{% else %}
desktop_base AS (
    SELECT
        "" as client_id, "" as segment, "" as branch
        {% for h in histograms %}, CAST(NULL AS STRUCT<sum INT64, values ARRAY<STRUCT<key STRING, value INT64>>>) as {{h.alias}}{% endfor %}
    WHERE FALSE
),
{% endif %}
{% if available_on_android == True %}
android_base AS (
    SELECT
        client_info.client_id as client_id,
        normalized_os as segment,
        mozfun.map.get_key(ping_info.experiments, "{{slug}}").branch as branch,
        {% for h in histograms %}
        f.{{h.name}} as {{h.alias}}{% if not forloop.last %},{% endif %}
        {% endfor %}
    FROM `mozdata.fenix.metrics` as f
    WHERE
      DATE(submission_timestamp) >= DATE('{{startDate}}')
      AND DATE(submission_timestamp) <= DATE('{{endDate}}')
      AND normalized_channel = "{{channel}}"
      {% if single_os_filter %}AND normalized_os = "{{single_os_filter}}"{% endif %}
      AND ARRAY_LENGTH(ping_info.experiments) > 0
      AND mozfun.map.get_key(ping_info.experiments, "{{slug}}").branch is not null
      AND (
        {% for h in histograms %}
        f.{{h.name}} is not null{% if not forloop.last %} OR {% endif %}
        {% endfor %}
      )
),
{% else %}
android_base AS (
    SELECT
        "" as client_id, "" as segment, "" as branch
        {% for h in histograms %}, CAST(NULL AS STRUCT<sum INT64, values ARRAY<STRUCT<key STRING, value INT64>>>) as {{h.alias}}{% endfor %}
    WHERE FALSE
),
{% endif %}
{% if include_non_enrolled_branch == True %}
{% if available_on_desktop == True %}
desktop_base_non_enrolled AS (
    SELECT
        client_info.client_id as client_id,
        normalized_os as segment,
        "default" as branch,
        {% for h in histograms %}
        d.{{h.name}} as {{h.alias}}{% if not forloop.last %},{% endif %}
        {% endfor %}
    FROM `mozdata.firefox_desktop.metrics` as d
    WHERE
      DATE(submission_timestamp) >= DATE('{{startDate}}')
      AND DATE(submission_timestamp) <= DATE('{{endDate}}')
      AND normalized_channel = "{{channel}}"
      AND normalized_app_name = "Firefox"
      {% if single_os_filter %}AND normalized_os = "{{single_os_filter}}"{% endif %}
      AND ARRAY_LENGTH(ping_info.experiments) = 0
      AND (
        {% for h in histograms %}
        d.{{h.name}} is not null{% if not forloop.last %} OR {% endif %}
        {% endfor %}
      )
),
{% else %}
desktop_base_non_enrolled AS (
    SELECT
        "" as client_id, "" as segment, "" as branch
        {% for h in histograms %}, CAST(NULL AS STRUCT<sum INT64, values ARRAY<STRUCT<key STRING, value INT64>>>) as {{h.alias}}{% endfor %}
    WHERE FALSE
),
{% endif %}
{% if available_on_android == True %}
android_base_non_enrolled AS (
    SELECT
        client_info.client_id as client_id,
        normalized_os as segment,
        "default" as branch,
        {% for h in histograms %}
        f.{{h.name}} as {{h.alias}}{% if not forloop.last %},{% endif %}
        {% endfor %}
    FROM `mozdata.fenix.metrics` as f
    WHERE
      DATE(submission_timestamp) >= DATE('{{startDate}}')
      AND DATE(submission_timestamp) <= DATE('{{endDate}}')
      AND normalized_channel = "{{channel}}"
      {% if single_os_filter %}AND normalized_os = "{{single_os_filter}}"{% endif %}
      AND ARRAY_LENGTH(ping_info.experiments) = 0
      AND (
        {% for h in histograms %}
        f.{{h.name}} is not null{% if not forloop.last %} OR {% endif %}
        {% endfor %}
      )
),
{% else %}
android_base_non_enrolled AS (
    SELECT
        "" as client_id, "" as segment, "" as branch
        {% for h in histograms %}, CAST(NULL AS STRUCT<sum INT64, values ARRAY<STRUCT<key STRING, value INT64>>>) as {{h.alias}}{% endfor %}
    WHERE FALSE
),
{% endif %}
{% endif %}
{% for h in histograms %}
{{h.alias}}_desktop_data AS (
    SELECT
        client_id, segment, branch,
        {% if h.distribution_type == "timing_distribution" %}
        CAST(CAST(key as INT64)/1000000 AS FLOAT64) AS bucket,
        {% else %}
        CAST(CAST(key as INT64) AS FLOAT64) AS bucket,
        {% endif %}
        value as count
    FROM desktop_base, UNNEST({{h.alias}}.values)
    WHERE {{h.alias}} is not null
),
{{h.alias}}_android_data AS (
    SELECT
        client_id, segment, branch,
        {% if h.distribution_type == "timing_distribution" %}
        CAST(CAST(key as INT64)/1000000 AS FLOAT64) AS bucket,
        {% else %}
        CAST(CAST(key as INT64) AS FLOAT64) AS bucket,
        {% endif %}
        value as count
    FROM android_base, UNNEST({{h.alias}}.values)
    WHERE {{h.alias}} is not null
),
{% if include_non_enrolled_branch == True %}
{{h.alias}}_desktop_data_non_enrolled AS (
    SELECT
        client_id, segment, branch,
        {% if h.distribution_type == "timing_distribution" %}
        CAST(CAST(key as INT64)/1000000 AS FLOAT64) AS bucket,
        {% else %}
        CAST(CAST(key as INT64) AS FLOAT64) AS bucket,
        {% endif %}
        value as count
    FROM desktop_base_non_enrolled, UNNEST({{h.alias}}.values)
    WHERE {{h.alias}} is not null
),
{{h.alias}}_android_data_non_enrolled AS (
    SELECT
        client_id, segment, branch,
        {% if h.distribution_type == "timing_distribution" %}
        CAST(CAST(key as INT64)/1000000 AS FLOAT64) AS bucket,
        {% else %}
        CAST(CAST(key as INT64) AS FLOAT64) AS bucket,
        {% endif %}
        value as count
    FROM android_base_non_enrolled, UNNEST({{h.alias}}.values)
    WHERE {{h.alias}} is not null
),
{% endif %}
{{h.alias}}_client_buckets AS (
    SELECT
        client_id, segment, branch, bucket,
        SUM(CAST(count AS NUMERIC)) as counts
    FROM (
        SELECT * FROM {{h.alias}}_desktop_data
        UNION ALL
        SELECT * FROM {{h.alias}}_android_data
        {% if include_non_enrolled_branch == True %}
        UNION ALL
        SELECT * FROM {{h.alias}}_desktop_data_non_enrolled
        UNION ALL
        SELECT * FROM {{h.alias}}_android_data_non_enrolled
        {% endif %}
    ) s
    GROUP BY client_id, segment, branch, bucket
),
{{h.alias}}_result AS (
    SELECT
        "{{h.name}}" as metric_name,
        client_id,
        segment,
        branch,
        ARRAY_AGG(STRUCT(bucket, counts) ORDER BY bucket) as bucket_counts
    FROM {{h.alias}}_client_buckets
    GROUP BY client_id, segment, branch
){% if not forloop.last %},{% endif %}
{% endfor %}
SELECT * FROM (
{% for h in histograms %}
    SELECT * FROM {{h.alias}}_result{% if not forloop.last %}
    UNION ALL{% endif %}
{% endfor %}
)
{% endautoescape %}
