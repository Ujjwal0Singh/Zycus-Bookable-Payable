# The Bookable Payable — Automated AP Decomposition & ERP Reconciliation

This repository contains our solution for **The Bookable Payable** engineering challenge. It processes raw PDF documents from `documents/`, segments multi-document packets, declines non-payables, resolves reference codes against `master_data/`, and runs an iterative, signal-driven reconciliation loop against `erp.py` to emit verified autodrafts in `output/`.

## Setup & Installation

1. **Install Python dependencies** (Python 3.10+):
   ```bash
   pip install -r requirements.txt
```
### 2. For Running the 
```bash
python run.py
```