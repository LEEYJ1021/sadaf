# Data Access Instructions

## Source File
`3월성과데이터(샘플).xlsx` — hourly Naver advertising performance data (March 2025, single advertiser, 89,675 rows × 16 columns).

## Availability
This dataset is proprietary and **not included** in this repository due to commercial confidentiality. The raw data is internal advertising-operations data maintained by **SearchM Co., Ltd.**, an official Naver and Kakao advertising agency, and cannot be redistributed publicly.

## Requesting Access
Data access requests are handled directly by SearchM Co., Ltd., not through this repository.

Please submit your request via the SearchM website: **https://www.searchm.co.kr/**

When requesting, please include:
1. Institutional affiliation and research purpose
2. Brief description of intended use
3. Confirmation the data will not be used commercially

Requests are evaluated case-by-case, and data may be provided **if there is a reasonable justification** for the request. Please note that access is not guaranteed and is granted at SearchM's discretion.

## Schema
See main README §"Column Schema" for the full 16-column specification (Date, Hours, customer_id, campaign_id, ad_group_id, ad_id, impression, click, cost, sum_of_ad_rank, conversion_count, sales_by_conversion, CTR, CVR, ROAS, Depth) and derived features (CPC, CPA, has_conversion, log_* transforms, hour_sin/cos, campaign_type).

## Placement
Once obtained, place the file at `data/3월성과데이터(샘플).xlsx`. All scripts in `scripts/` reference this path via `--data_path` argument (default: `data/3월성과데이터(샘플).xlsx`).
