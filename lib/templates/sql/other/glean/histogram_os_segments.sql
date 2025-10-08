{% autoescape off %}
-- Unified histogram query template for non-experiments
-- Supports both custom conditions and standard conditions with OS segments only

{% if prerequisite_ctes %}
WITH
{{ prerequisite_ctes }},
{% else %}
WITH
{% endif %}
{% for branch in branches %}
{% if needs_desktop %}
{{branch.safe_name}}_desktop as (
    SELECT
        normalized_os as segment,
        "{{branch.name}}" as branch,
        {% if distribution_type == "timing_distribution" %}
        CAST(key as INT64)/1000000 AS bucket,
        {% else %}
        CAST(key as INT64) AS bucket,
        {% endif %}
        value as count
    FROM `mozdata.firefox_desktop.metrics` as m
        CROSS JOIN UNNEST(m.{{histogram}}.values)
    WHERE
        {% if use_shared_dates %}
        DATE(submission_timestamp) >= DATE('{{start_date}}')
        AND DATE(submission_timestamp) <= DATE('{{end_date}}')
        AND normalized_channel = "{{channel}}"
        {% else %}
        DATE(submission_timestamp) >= DATE('{{branch.startDate}}')
        AND DATE(submission_timestamp) <= DATE('{{branch.endDate}}')
        AND normalized_channel = "{{branch.channel}}"
        {% endif %}
        AND normalized_app_name = "Firefox"
        {% if sample_threshold %}AND MOD(ABS(FARM_FINGERPRINT(document_id)), {{sample_modulus}}) <= {{sample_threshold}}{% endif %}
        AND m.{{histogram}} is not null
        {% if branch.custom_condition %}
        AND ({{branch.custom_condition}})
        {% endif %}
        {% if branch.ver_condition %}{{branch.ver_condition}}{% endif %}
        {% if branch.arch_condition %}{{branch.arch_condition}}{% endif %}
        {% if branch.glean_conditions %}
        {% for condition in branch.glean_conditions %}
        {{condition}}
        {% endfor %}
        {% endif %}
){% if needs_android %},{% elif not forloop.last %},{% endif %}
{% endif %}
{% if needs_android %}
{{branch.safe_name}}_android as (
    SELECT
        normalized_os as segment,
        "{{branch.name}}" as branch,
        {% if distribution_type == "timing_distribution" %}
        CAST(key as INT64)/1000000 AS bucket,
        {% else %}
        CAST(key as INT64) AS bucket,
        {% endif %}
        value as count
    FROM `mozdata.fenix.metrics` as f
        CROSS JOIN UNNEST(f.{{histogram}}.values)
    WHERE
        {% if use_shared_dates %}
        DATE(submission_timestamp) >= DATE('{{start_date}}')
        AND DATE(submission_timestamp) <= DATE('{{end_date}}')
        AND normalized_channel = "{{channel}}"
        {% else %}
        DATE(submission_timestamp) >= DATE('{{branch.startDate}}')
        AND DATE(submission_timestamp) <= DATE('{{branch.endDate}}')
        AND normalized_channel = "{{branch.channel}}"
        {% endif %}
        {% if sample_threshold %}AND MOD(ABS(FARM_FINGERPRINT(document_id)), {{sample_modulus}}) <= {{sample_threshold}}{% endif %}
        AND f.{{histogram}} is not null
        {% if branch.custom_condition_android %}
        AND ({{branch.custom_condition_android}})
        {% endif %}
        {% if branch.ver_condition %}{{branch.ver_condition}}{% endif %}
        {% if branch.arch_condition %}{{branch.arch_condition}}{% endif %}
        {% if branch.glean_conditions %}
        {% for condition in branch.glean_conditions %}
        {{condition}}
        {% endfor %}
        {% endif %}
){% if not forloop.last %},{% endif %}
{% endif %}
{% endfor %}

SELECT
    segment,
    branch,
    bucket,
    SUM(count) as counts
FROM
    (
{% for branch in branches %}
{% if needs_desktop %}
        SELECT * FROM {{branch.safe_name}}_desktop
{% if needs_android %}
        UNION ALL
        SELECT * FROM {{branch.safe_name}}_android
{% endif %}
{% elif needs_android %}
        SELECT * FROM {{branch.safe_name}}_android
{% endif %}
{% if not loop.last %}
        UNION ALL
{% endif %}
{% endfor %}
    ) s
GROUP BY
  segment, branch, bucket
ORDER BY
  segment, branch, bucket
{% endautoescape %}