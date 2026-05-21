# tapegif capture script for examples/demo_tui.py (use --fast via app_args)
size: [120, 35]
gif_width: 1000

app_args:
  fast: true

steps:
  - sleep: 1.5
    capture: 1600
  - sleep: 3.0
    capture: 2000
  - sleep: 4.0
    capture: 2200
  - sleep: 4.0
    capture: 2200
  - sleep: 5.0
    capture: 2800
  - sleep: 7.0
    capture: 3000
  - sleep: 6.0
    capture: 2800
  - sleep: 4.0
    capture: 2200
