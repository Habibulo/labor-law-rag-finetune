# Archived: 법제처 Open API client (not used)

`api_smoke_test.py` called the law.go.kr DRF endpoints (`lawSearch.do`, `lawService.do`) with an `OC` key.
It was never run: open.law.go.kr registration was not reachable (2026-09-25), so the project switched to the
legalize-kr GitHub mirrors, which are generated from the same official API. See docs/decisions_log.md #5.

Kept only as a record of the original design. Nothing in `src/` imports it.
