{% autoescape off %}
with 
{% for branch in branches %}
crashdata_{{branch.name}}_desktop as (
    SELECT
        normalized_os as segment,
        COUNT(*) as crash_count
    FROM
        `moz-fx-data-shared-prod.firefox_desktop.crash` as m
    WHERE
        DATE(submission_timestamp) >= DATE('{{branch.startDate}}')
        AND DATE(submission_timestamp) <= DATE('{{branch.endDate}}')
        AND normalized_channel = "{{branch.channel}}"
        AND normalized_app_name = "Firefox"
        {% if single_os_filter %}AND normalized_os = "{{single_os_filter}}"{% endif %}
        {{branch.ver_condition}}
        {{branch.arch_condition}}
{% for condition in branch.glean_conditions %}
        {{condition}}
{% endfor %}
    GROUP BY
        segment
),
crashdata_{{branch.name}}_android as (
    SELECT
        normalized_os as segment,
        COUNT(*) as crash_count
    FROM
        `moz-fx-data-shared-prod.fenix.crash` as m
    WHERE
        DATE(submission_timestamp) >= DATE('{{branch.startDate}}')
        AND DATE(submission_timestamp) <= DATE('{{branch.endDate}}')
        AND normalized_channel = "{{branch.channel}}"
        {% if single_os_filter %}AND normalized_os = "{{single_os_filter}}"{% endif %}
        {{branch.ver_condition}}
        {{branch.arch_condition}}
{% for condition in branch.glean_conditions %}
        {{condition}}
{% endfor %}
    GROUP BY
        segment
),
aggregate_{{branch.name}}_desktop as (
SELECT
    segment,
    "{{branch.name}}" as branch,
    crash_count
FROM
    crashdata_{{branch.name}}_desktop
),
aggregate_{{branch.name}}_android as (
SELECT
    segment,
    "{{branch.name}}" as branch,
    crash_count
FROM
    crashdata_{{branch.name}}_android
)
{% if branch.last == False %}
,
{% endif %}
{% endfor %}

SELECT
    segment,
    branch,
    crash_count
FROM
    (
{% for branch in branches %}
        SELECT * FROM aggregate_{{branch.name}}_desktop
        UNION ALL
        SELECT * FROM aggregate_{{branch.name}}_android
{% if branch.last == False %}
        UNION ALL
{% endif %}
{% endfor %}
    )
ORDER BY
    segment, branch
{% endautoescape %}