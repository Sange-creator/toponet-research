# Highway dashboard observations

This folder contains a privacy-filtered extract of three network-dashboard screen recordings made on an iPhone 15 during travel along the Prithvi Highway. The recordings were sampled at one-second intervals. The counts are 1,396, 609, and 3,059 timestamped rows (5,064 rows total).

The observations provide field context for the paper. They were not used to train, tune, or evaluate the simulated handover controller. They are dashboard readings, not operator logs or a field test of the controller.

## Files

- `highway_dashboard_readings.csv`: one row per sampled timestamp. `recording_id` distinguishes the three recordings, and `source_excel_row` maps a row to its position in the supplied workbook.

## Columns

The CSV retains the source workbook's elapsed time, carrier, network capability/type, TAC, PLMN, ARFCN, band, bandwidth, displayed cell ID, PCI, ECN0, LAC, RSCP, RSRP, RSRQ, SINR0, SINR1, SCN, screen status, review flags, QA status, and QA notes. The `Phone Number` and `Raw OCR Text` columns were removed. Source recording URLs and filenames are not included.

Blank values are kept as blank. In the source workbook, a blank can mean that a value was not visible, did not apply, or could not be read. Values were not carried forward from adjacent samples. The status counts in the source workbook are:

| Recording | Samples | Dashboard visible | Partial | Dashboard not visible |
| --- | ---: | ---: | ---: | ---: |
| 1 | 1,396 | 725 | 1 | 670 |
| 2 | 609 | 605 | 4 | 0 |
| 3 | 3,059 | 3,056 | 3 | 0 |
| **Total** | **5,064** | **4,386** | **8** | **670** |

The QA status and notes are retained from the source workbook. They record second-pass or partial-source review, but do not certify that every value was independently verified.

## Limits and privacy

The workbook does not include GPS coordinates or the exact start point, end point, or direction of travel. The Prithvi Highway label and iPhone 15 model are provided by the authors. The source recordings are not included. The displayed cell ID and signal readings can show observed radio variation; a cell-ID change alone does not establish a handover, and these data do not measure throughput or controller performance.

This release removes the phone-number and raw-OCR fields but retains radio measurements and displayed cell identifiers. No reuse license has been assigned to this dataset in this release.
