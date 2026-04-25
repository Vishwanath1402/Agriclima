---
title: AgriClima
emoji: 🌱
colorFrom: green
colorTo: yellow
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: apache-2.0
---

# 🌱 AgriClima — A Climate-Resilient Farming Advisor

**Built for the Gemma 4 Good Hackathon (April–May 2026)**

## The problem

Over 500 million smallholder farmers worldwide are on the front line of climate
change. Rainfall patterns are shifting, growing seasons are changing, and crop
varieties that worked for generations are starting to fail. Most of these
farmers do not have reliable access to agricultural extension officers,
internet-based advice services, or up-to-date climate-adaptation guidance.

## The approach

AgriClima is a simple, focused application that gives smallholder farmers
practical advice for adapting to changing climate conditions in their region.
A farmer (or a community worker assisting them) provides four pieces of
information:

1. Their region
2. The crop they grow or plan to grow
3. A recent climate or weather observation in their own words
4. Their specific question or concern

The system uses **Gemma 4** — Google's open-weight language model family — to
return:

- A short risk assessment
- A few concrete recommendations
- One actionable thing to do this week

## Why Gemma 4?

Gemma 4 is the right model for this problem because it is **open-weight** and
runs efficiently on consumer-grade hardware. This means AgriClima can be
deployed in environments without reliable internet — exactly the settings where
smallholder farmers most often operate.

The current demo uses the `gemma-4-e4b-it` variant (an "edge" model designed
for resource-constrained settings).

## How to run locally

```bash
pip install -r requirements.txt
python app.py
```

Then open http://localhost:7860 in your browser.

## Project structure

```
.
├── app.py             # Gradio app + Gemma 4 inference
├── requirements.txt   # Python dependencies
└── README.md          # this file
```

## Limitations and honest disclaimers

- AgriClima provides general guidance only. It is not a substitute for local
  agronomic expertise. Major farming decisions should be confirmed with local
  extension officers when possible.
- The model has not been fine-tuned on region-specific agricultural data; it
  relies on Gemma 4's general knowledge.
- A real-world deployment would benefit from: integration with local weather
  data, region-specific crop databases, multilingual support, and offline-first
  packaging for low-end Android devices.

## License

Apache 2.0
