# VSL400 Access Preparation

VSL400 is a controlled-access Vietnamese Sign Language video dataset. Do not
attempt to bypass access controls.

## Official Record

- Dataset DOI: `https://doi.org/10.5281/zenodo.17943574`
- Data Usage Agreement record: `https://zenodo.org/records/21366957`
- Access type: controlled access with Data Usage Agreement.

The public description reports a multi-view isolated VSL dataset with three
camera folders:

```text
VSL400/
  front_view/
  left_view/
  right_view/
  front_view.json
  left_view.json
  right_view.json
```

This layout is based on public documentation only. VietSignAI must inspect the
actual authorized local files before finalizing parser assumptions.

## Expected Local Path

After approval and authorized download, place files under:

```text
data/raw/vsl400/
```

Then run:

```bash
python -m vietsign.cli dataset verify vsl400
```

Current adapter status without local files:

```text
WAITING_FOR_AUTHORIZED_DATA
```

## Policy

- Do not commit VSL400 videos or metadata copied from a controlled archive.
- Do not automate access bypass.
- Do not claim VSL400 support has been real-tested until authorized files are
  inspected locally.
