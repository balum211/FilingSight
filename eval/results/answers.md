# Answer evaluation (extractive (no LLM), semantic chunks, local reranker)

| Metric | Value |
|---|---|
| Unanswerable questions correctly refused | 100% (5) |
| Answerable questions wrongly refused | 0% (20) |
| Answered with the gold evidence in a cited source | 90% |
| Latency p50 / p95 | 3.2s / 5.9s |

| Q | Route | Cited evidence | Faithful | Correct | Latency | Answer |
|---|---|---|---|---|---|---|
| U01 | answered | True |  |  | 6.2s | Most relevant passages from the filings (no LLM key set, so this is extractive): - Year Ended December 31, / 2019 / 2020 / 2021 › Mobility revenue / $10,707 / $6,089 / $6,953 [2] - 2020 to 2021 % Change › Revenue / $11,1 |
| U02 | answered | True |  |  | 3.2s | Most relevant passages from the filings (no LLM key set, so this is extractive): - 2020 to 2021 % Change › Monthly Active Platform Consumers ("MAPCs") / 93 / 118 / 27 % [1] - • our expectations regarding future operating |
| U03 | answered | True |  |  | 3.6s | Most relevant passages from the filings (no LLM key set, so this is extractive): - Costs and expenses › Net loss attributable to Uber Technologies, Inc. / (61)% / (3)% [2] - Net loss attributable to Uber Technologies, In |
| U04 | answered | True |  |  | 3.0s | Most relevant passages from the filings (no LLM key set, so this is extractive): - Year Ended December 31, / 2019 / 2020 / 2021 › Mobility revenue / $10,707 / $6,089 / $6,953 [3] - We are a global company and as of Decem |
| U05 | answered | True |  |  | 3.2s | Most relevant passages from the filings (no LLM key set, so this is extractive): - 2020 to 2021 % Change › Mobility Adjusted EBITDA / $1,169 / $1,596 / 37 % [2] - 2020 to 2021 % Change › Adjusted EBITDA / $(2,528) / $(77 |
| U06 | answered | True |  |  | 2.9s | Most relevant passages from the filings (no LLM key set, so this is extractive): - On November 12, 2021, we completed the acquisition of Transplace in an all-cash transaction, allowing us to expand our Uber Freight busin |
| U07 | answered | True |  |  | 3.0s | Most relevant passages from the filings (no LLM key set, so this is extractive): - In 2021, 74% of our Gross Bookings were paid by either credit card or debit card. [1] - Overall Gross Bookings increased by $32.5 billion |
| U08 | answered | True |  |  | 3.2s | Most relevant passages from the filings (no LLM key set, so this is extractive): - In 2021, we derived 23% of our Mobility Gross Bookings from five metropolitan areas—Chicago, Miami, and New York City in the United State |
| U09 | answered | True |  |  | 3.5s | Most relevant passages from the filings (no LLM key set, so this is extractive): - We were founded in 2009 and incorporated as Ubercab, Inc., a Delaware corporation, in July 2010. [1] - ("Uber," "we," "our," or "us") was |
| L01 | answered | True |  |  | 3.3s | Most relevant passages from the filings (no LLM key set, so this is extractive): - For the years ended December 31, 2021, 2020 and 2019, in relation to the driver, rider, Light Vehicle riders and Lyft Rentals renters inc |
| L02 | answered | True |  |  | 3.0s | Most relevant passages from the filings (no LLM key set, so this is extractive): - Revenue per Active Rider represents our ability to drive usage and monetization of our platform. › Active Riders / 2021 / 2020 / 2019 [4] |
| L03 | answered | True |  |  | 3.4s | Most relevant passages from the filings (no LLM key set, so this is extractive): - Adjusted EBITDA in 2021 was $92.9 million, marking our first annual Adjusted EBITDA profitability. [1] - The following table provides a r |
| L04 | answered | True |  |  | 2.9s | Most relevant passages from the filings (no LLM key set, so this is extractive): - On July 13, 2021, the Company completed a multi-element transaction with Woven Planet, a subsidiary of Toyota Motor Corporation, for the  |
| L05 | answered | True |  |  | 2.7s | Most relevant passages from the filings (no LLM key set, so this is extractive): - The number of our full-time employees increased from 2,708 as of December 31, 2017, to 4,453 as of December 31, 2021. [1] - As of Decembe |
| L06 | answered | True |  |  | 3.7s | Most relevant passages from the filings (no LLM key set, so this is extractive): - Cash used in operating activities was $101.7 million for the year ended December 31, 2021. [1] - Cash used in operating activities was $1 |
| L07 | answered | True |  |  | 2.7s | Most relevant passages from the filings (no LLM key set, so this is extractive): - We founded Lyft in 2012 and changed our name to Lyft, Inc. in 2013 when we sold the assets related to our Zimride operations. [1] - (the  |
| C01 | answered | False |  |  | 5.9s | Most relevant passages from the filings (no LLM key set, so this is extractive): - **Uber:** For the year ended December 31, 2021 compared to the same period in 2020, Mobility revenue increased $864 million, or 14% and M |
| C02 | answered | True |  |  | 4.5s | Most relevant passages from the filings (no LLM key set, so this is extractive): - **Uber:** In California, we welcomed the passage of Proposition 22, which introduced new requirements for platform companies like Uber to |
| C03 | answered | True |  |  | 5.8s | Most relevant passages from the filings (no LLM key set, so this is extractive): - **Uber:** 2020 to 2021 % Change › Mobility Adjusted EBITDA / $1,169 / $1,596 / 37 % [3] - **Lyft:** Adjusted EBITDA in 2021 was $92.9 mil |
| A01 | answered | False |  |  | 5.6s | Most relevant passages from the filings (no LLM key set, so this is extractive): - **Uber:** As of December 31, 2021, we also had foreign net operating loss carryforwards of $10.6 billion, of which $507 million will begi |
| X01 | refused |  |  |  | 0.0s | I could not find this in the Uber and Lyft FY2021 10-K filings. |
| X02 | refused |  |  |  | 4.0s | I could not find this in the Uber and Lyft FY2021 10-K filings. |
| X03 | refused |  |  |  | 1.9s | I could not find this in the Uber and Lyft FY2021 10-K filings. |
| X04 | refused |  |  |  | 4.1s | I could not find this in the Uber and Lyft FY2021 10-K filings. |
| X05 | refused |  |  |  | 2.5s | I could not find this in the Uber and Lyft FY2021 10-K filings. |
