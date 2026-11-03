{% autoescape off %}
WITH
desktop_eventdata AS (
SELECT
  document_id,
  normalized_os as segment,
  mozfun.map.get_key(ping_info.experiments, "{{slug}}").branch as branch,
  SAFE_CAST((SELECT value FROM UNNEST(event.extra) WHERE key = '{{metric}}') AS int) AS bucket
FROM
  `moz-fx-data-shared-prod.firefox_desktop.pageload` as d
CROSS JOIN
  UNNEST(events) AS event
WHERE
  normalized_channel = "{{channel}}"
  AND DATE(submission_timestamp) >= DATE('{{startDate}}')
  AND DATE(submission_timestamp) <= DATE('{{endDate}}')
  {% if single_os_filter %}AND normalized_os = "{{single_os_filter}}"{% endif %}
  AND mozfun.map.get_key(ping_info.experiments, "{{slug}}").branch is not null
  {% if pageload_event_filter %}
  AND {{pageload_event_filter}}
  {% endif %}
)
{% if include_non_enrolled_branch == True %}
,
desktop_eventdata_non_enrolled AS (
SELECT
  document_id,
  normalized_os as segment,
  "non-enrolled" as branch,
  SAFE_CAST((SELECT value FROM UNNEST(event.extra) WHERE key = '{{metric}}') AS int) AS bucket
FROM
  `moz-fx-data-shared-prod.firefox_desktop.pageload`
CROSS JOIN
  UNNEST(events) AS event
WHERE
  normalized_channel = "{{channel}}"
  AND DATE(submission_timestamp) >= DATE('{{startDate}}')
  AND DATE(submission_timestamp) <= DATE('{{endDate}}')
  {% if single_os_filter %}AND normalized_os = "{{single_os_filter}}"{% endif %}
  AND ARRAY_LENGTH(ping_info.experiments) = 0
  {% if pageload_event_filter %}
  AND {{pageload_event_filter}}
  {% endif %}
)
{% endif %}
, android_eventdata AS (
SELECT
  document_id,
  normalized_os as segment,
  mozfun.map.get_key(ping_info.experiments, "{{slug}}").branch as branch,
  SAFE_CAST((SELECT value FROM UNNEST(event.extra) WHERE key = '{{metric}}') AS int) AS bucket
FROM
  `moz-fx-data-shared-prod.fenix.pageload` as f
CROSS JOIN
  UNNEST(events) AS event
WHERE
  normalized_channel = "{{channel}}"
  AND DATE(submission_timestamp) >= DATE('{{startDate}}')
  AND DATE(submission_timestamp) <= DATE('{{endDate}}')
  {% if single_os_filter %}AND normalized_os = "{{single_os_filter}}"{% endif %}
  AND mozfun.map.get_key(ping_info.experiments, "{{slug}}").branch is not null
  {% if pageload_event_filter %}
  AND {{pageload_event_filter}}
  {% endif %}
)
{% if include_non_enrolled_branch == True %}
,
android_eventdata_non_enrolled AS (
SELECT
  document_id,
  normalized_os as segment,
  "non-enrolled" as branch,
  SAFE_CAST((SELECT value FROM UNNEST(event.extra) WHERE key = '{{metric}}') AS int) AS bucket
FROM
  `moz-fx-data-shared-prod.fenix.pageload`
CROSS JOIN
  UNNEST(events) AS event
WHERE
  normalized_channel = "{{channel}}"
  AND DATE(submission_timestamp) >= DATE('{{startDate}}')
  AND DATE(submission_timestamp) <= DATE('{{endDate}}')
  {% if single_os_filter %}AND normalized_os = "{{single_os_filter}}"{% endif %}
  AND ARRAY_LENGTH(ping_info.experiments) = 0
  {% if pageload_event_filter %}
  AND {{pageload_event_filter}}
  {% endif %}
)
{% endif %}
, all_events AS (
  SELECT * FROM desktop_eventdata
  {% if include_non_enrolled_branch == True %}
  UNION ALL
  SELECT * FROM desktop_eventdata_non_enrolled
  {% endif %}
  UNION ALL
  SELECT * FROM android_eventdata
  {% if include_non_enrolled_branch == True %}
  UNION ALL
  SELECT * FROM android_eventdata_non_enrolled
  {% endif %}
)
, document_buckets AS (
  SELECT
    document_id,
    segment,
    branch,
    bucket,
    COUNT(*) as counts
  FROM all_events
  WHERE bucket >= {{minVal}} AND bucket <= {{maxVal}}
  GROUP BY document_id, segment, branch, bucket
)
SELECT
  document_id,
  segment,
  branch,
  ARRAY_AGG(STRUCT(bucket, counts) ORDER BY bucket) as bucket_counts
FROM document_buckets
GROUP BY document_id, segment, branch
{% endautoescape %}
