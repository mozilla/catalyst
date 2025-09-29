{% autoescape off %}
-- Custom segment pageload events query template
-- Supports custom user segments with arbitrary SQL conditions

{% if prerequisite_ctes %}
{{ prerequisite_ctes }},
{% endif %}

{% for branch in branches %}
{% for segment in segments %}
{{branch.name}}_{{segment.name}}_desktop as (
    SELECT
        "{{segment.name}}" as segment,
        "{{branch.name}}" as branch,
{% for metric in metrics %}
        {{metric}} as {{metric}}{% if not loop.last %},{% endif %}
{% endfor %}
    FROM `mozdata.firefox_desktop.events` as m
    WHERE
        DATE(submission_timestamp) >= DATE('{{branch.startDate}}')
        AND DATE(submission_timestamp) <= DATE('{{branch.endDate}}')
        AND normalized_channel = "{{branch.channel}}"
        AND normalized_app_name = "Firefox"
        AND event_category = 'navigation'
        AND event_name = 'load'
        {{branch.ver_condition}}
        {{branch.arch_condition}}
{% for condition in branch.glean_conditions %}
        {{condition}}
{% endfor %}
{% for condition in segment.conditions %}
        AND ({{condition}})
{% endfor %}
        -- Filter out extreme outliers
{% for metric in metrics %}
        AND {{metric}} IS NOT NULL
        AND {{metric}} > 0
        AND {{metric}} < {{metrics[metric].max}}
{% endfor %}
){% if not (branch.last and segment.last) %},{% endif %}
{% endfor %}
{% endfor %}

SELECT
    segment,
    branch,
{% for metric in metrics %}
    {{metric}}{% if not loop.last %},{% endif %}
{% endfor %}
FROM
    (
{% for branch in branches %}
{% for segment in segments %}
        SELECT * FROM {{branch.name}}_{{segment.name}}_desktop{% if not (branch.last and segment.last) %}
        UNION ALL{% endif %}
{% endfor %}
{% endfor %}
    )
ORDER BY
    segment, branch{% for metric in metrics %}, {{metric}}{% endfor %}
{% endautoescape %}