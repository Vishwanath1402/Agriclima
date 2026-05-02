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
change. Rainfall patterns are shifting, growing seasons are arriving at unfamiliar
times, and crop varieties that worked for generations are starting to fail. Most
of these farmers do not have reliable access to agricultural extension officers,
internet-based advice services, or up-to-date climate-adaptation guidance.

## The approach

AgriClima is a focused application that gives smallholder farmers practical
advice for adapting to changing climate conditions in their region. The app uses
**Gemma 4** — Google's open-weight language model — running with simple
infrastructure suited to low-resource environments.

A farmer (or a community worker assisting them) describes their situation:
their region, the crop they grow, what they have observed about local weather
or climate patterns, and their specific question or concern. AgriClima then
returns:

- A short risk assessment
- A few concrete recommendations specific to the region
- One actionable thing to do this week

## Two ways to use the app

The interface has two tabs:

1. **Quick consultation** (default) — a structured form with four fields. Best
   for getting a complete advisory in a single response. The structured input
   produces the most reliable output.

2. **Continue conversation** — a chat interface for follow-up questions. After
   submitting a quick consultation, click "Continue in conversation" to carry
   the same context into a back-and-forth conversation.

## Why Gemma 4?

Gemma 4 is the right model for this problem because it is **open-weight** and
runs efficiently on consumer-grade hardware. This means AgriClima can be
deployed in environments without reliable internet — exactly the settings where
smallholder farmers most often operate. The current demo uses the
`gemma-4-e4b-it` variant, an "edge" model designed for resource-constrained
settings.

## Design choices worth flagging

A few choices that shape how the system behaves:

**1. Honest cultivar handling.** When the model is not confident about specific
crop varieties for a region, it tells the farmer plainly and points them to a
named local research institution (e.g., KVK in India, KALRO in Kenya, EMBRAPA
in Brazil, CSIRO in Australia, IRRI for rice). This honesty is more useful than
a confident-sounding hedge.

**2. Clarification over guessing.** If the user's input is too vague to give
region-specific advice, the model asks 2–3 specific follow-up questions instead
of producing generic filler.

**3. Refuses harmful requests.** The system declines to help with questions
that would put consumers or the environment at risk (e.g., evading food safety
checks). It redirects toward legitimate agricultural goals where possible.

**4. Stays in scope.** Off-topic queries (financial advice, general chat) are
politely redirected back to farming.

**5. Generation temperature set to 0.5.** This is a deliberate
middle-ground choice for an advisory context, validated empirically.
At the typical chatbot default of 0.7, key features like institutional
referrals (KVK, KALRO, EMBRAPA, IRRI, CSIRO) appeared inconsistently —
roughly 1 in 3 runs of the same query. At 0.5, the same query produces
the institutional referral consistently. Empirical research on LLM
temperature settings (Renze 2024; document Q&A studies, 2026) shows
that temperature has less impact on factual accuracy than commonly
believed, but does meaningfully affect output consistency. The setting
is lower than typical chatbot defaults (0.7-0.9) for advisory consistency,
but higher than greedy decoding (0.0-0.3) to retain natural conversational
variation. Production AI advisory systems in agriculture and healthcare
typically operate in the 0.5-0.7 range; AgriClima sits at the lower end
given the higher-stakes nature of crop and climate advice.

## How to run locally

```bash
pip install -r requirements.txt
python app.py
```

Then open the URL printed in the terminal (typically http://localhost:7860).

## Project structure

```
.
├── app.py             # Gradio app + Gemma 4 inference
├── requirements.txt   # Python dependencies
├── README.md          # this file
└── LICENSE            # Apache 2.0
```

## Limitations and honest disclaimers

AgriClima provides general guidance only. It is not a substitute for local
agronomic expertise. Major farming decisions should be confirmed with local
extension officers when possible.

A few specific limitations worth being transparent about:

- **Cultivar specificity varies by region.** The model has stronger knowledge
  of well-documented agricultural regions (e.g., parts of South Asia) than
  others. The system is designed to be honest about this — it points farmers
  to local institutions when it is unsure.

- **Response quality varies across runs.** Like all language models, AgriClima
  is probabilistic and produces slightly different responses each time. In
  production, this would be addressed through ensembling, lower-temperature
  generation for high-stakes outputs, or human-in-the-loop review.

- **Not fine-tuned on agricultural corpora.** AgriClima uses Gemma 4 directly,
  relying on its general training. A fine-tuned version trained on agronomic
  literature, regional climate data, and verified extension content would
  produce more authoritative output.

## Future work

A real-world deployment of this concept would benefit from:

- Integration with local weather data and seasonal forecasts
- Multilingual support for major farmer languages (Hindi, Swahili, Portuguese,
  Vietnamese, Arabic, etc.)
- Region-specific crop databases for verified cultivar recommendations
- Offline-first packaging for low-end Android devices
- A voice interface for farmers with low literacy
- Fine-tuning on verified agricultural extension content

## License

Apache 2.0

