{% autoescape off %}
with desktop_crashdata as (
SELECT
  normalized_os as segment,
  mozfun.map.get_key(ping_info.experiments, "{{slug}}").branch as branch,
  COUNT(*) as crash_count
FROM
  `moz-fx-data-shared-prod.firefox_desktop.crash` as d
WHERE
  normalized_channel = "{{channel}}"
  AND DATE(submission_timestamp) >= DATE('{{startDate}}')
  AND DATE(submission_timestamp) <= DATE('{{endDate}}')  
  AND mozfun.map.get_key(ping_info.experiments, "{{slug}}").branch is not null
GROUP BY
  segment, branch
)
{% if include_non_enrolled_branch == True %}
,
desktop_crashdata_non_enrolled as (
SELECT
  normalized_os as segment,
  "non-enrolled" as branch,
  COUNT(*) as crash_count
FROM
  `moz-fx-data-shared-prod.firefox_desktop.crash`
WHERE
  normalized_channel = "{{channel}}"
  AND DATE(submission_timestamp) >= DATE('{{startDate}}')
  AND DATE(submission_timestamp) <= DATE('{{endDate}}')
  AND ARRAY_LENGTH(ping_info.experiments) = 0
GROUP BY
  segment, branch
)
{% endif %}
, android_crashdata as (
SELECT
  normalized_os as segment,
  mozfun.map.get_key(ping_info.experiments, "{{slug}}").branch as branch,
  COUNT(*) as crash_count
FROM
  `moz-fx-data-shared-prod.fenix.crash` as f
WHERE
  normalized_channel = "{{channel}}"
  AND DATE(submission_timestamp) >= DATE('{{startDate}}')
  AND DATE(submission_timestamp) <= DATE('{{endDate}}')  
  AND mozfun.map.get_key(ping_info.experiments, "{{slug}}").branch is not null
GROUP BY
  segment, branch
)
{% if include_non_enrolled_branch == True %}
,
android_crashdata_non_enrolled as (
SELECT
  normalized_os as segment,
  "non-enrolled" as branch,
  COUNT(*) as crash_count
FROM
  `moz-fx-data-shared-prod.fenix.crash`
WHERE
  normalized_channel = "{{channel}}"
  AND DATE(submission_timestamp) >= DATE('{{startDate}}')
  AND DATE(submission_timestamp) <= DATE('{{endDate}}')
  AND ARRAY_LENGTH(ping_info.experiments) = 0
GROUP BY
  segment, branch
)
{% endif %}

SELECT
  segment,
  branch,
  crash_count
FROM
{% if include_non_enrolled_branch == True %}
  (
    SELECT * from desktop_crashdata
    UNION ALL
    SELECT * from desktop_crashdata_non_enrolled
    UNION ALL
    SELECT * from android_crashdata
    UNION ALL
    SELECT * from android_crashdata_non_enrolled
  )
{% else %}
  (
    SELECT * from desktop_crashdata
    UNION ALL
    SELECT * from android_crashdata
  )
{% endif %}
ORDER BY
  segment, branch
{% endautoescape %}