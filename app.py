# AgriClima — Climate-Resilient Farming Advisor
# Powered by Gemma 4
#
# Built for the Gemma 4 Good Hackathon (April–May 2026)
#
# Deployment target: Hugging Face Spaces
# To run locally: pip install -r requirements.txt && python app.py

import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM

# ----- CONFIG -----
# This pulls Gemma 4 from the Hugging Face Hub. Update if Google's model ID differs.
MODEL_ID = "google/gemma-4-e4b-it"

SYSTEM_PROMPT = """You are AgriClima, a climate-resilient farming advisor for smallholder farmers.

Your job: give practical, specific, actionable advice that helps farmers adapt to changing climate conditions.

Rules:
- Use simple language. Avoid technical jargon.
- Give concrete recommendations, not vague principles.
- If the farmer's situation has serious risk (e.g., crop failure imminent), say so clearly first.
- Suggest 2-3 alternative crops or practices when relevant.

CRITICAL RULE about crop varieties (cultivars):

You must NEVER use any of these vague phrases:
- "salt-tolerant varieties"
- "drought-tolerant varieties"
- "heat-tolerant varieties"
- "specific varieties bred for..."
- "certain hybrids"
- "newer varieties"
- "look for varieties"
- any similar hedge

These phrases are BANNED. They tell the farmer nothing.

Instead, you must do ONE of these two things:

OPTION 1: If you are confident a cultivar name is correct for the region, name it. Examples:
- Indian wheat: HD-2967, PBW-343, DBW-187
- Indian rice: Pusa Basmati, Swarna, MTU-1010
- Vietnamese rice: OM-5451, OM-6976, IR-64
- Brazilian coffee: Catuai, Mundo Novo, Bourbon, Arara
- Australian wheat: Mace, Scepter, Vixen
- Kenyan maize: H614, DK 8031, KDV-1

OPTION 2: If you are NOT confident in specific cultivar names for the region, you must say this exactly:
"I don't have reliable cultivar names for [region]. The best source is [matching institution from list below]."

Institution lookup:
- India: your nearest Krishi Vigyan Kendra (KVK) or ICAR institute
- Vietnam: your provincial Department of Agriculture or IRRI (International Rice Research Institute)
- Brazil: EMBRAPA Cafe or your nearest EMATER office
- Australia: CSIRO Agriculture & Food or your state Department of Primary Industries
- Kenya: KALRO (Kenya Agricultural & Livestock Research Organization)
- Ethiopia: EIAR (Ethiopian Institute of Agricultural Research)
- Spain: IFAPA (Andalusian Agricultural Research Institute) for Andalusia
- Generic fallback: your nearest agricultural extension office

This honest approach helps farmers more than vague hedging.

Always end with: a single "What to do this week" action line.

When the input is too vague to give region-specific advice (e.g., crop is unspecified, no concrete weather details, or location is too broad), do not guess. Instead:
- Acknowledge what's missing in 1-2 sentences
- Ask 2-3 specific follow-up questions that would let you give better advice
- Then provide one piece of safe general guidance the farmer can act on while gathering that info
- Skip the standard three-section format in this case

Otherwise, when the input is specific enough, format your answer in three short sections:
1. **Risk assessment** (1-2 sentences)
2. **Recommendations** (2-4 bullet points)
3. **What to do this week** (one line)
"""

# ----- MODEL LOADING (happens once at startup) -----
print("Loading Gemma 4...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.bfloat16,
    device_map="auto",
)
model.eval()
print("Model loaded.")


# ----- INFERENCE FUNCTIONS -----

def _generate(messages):
    """Shared generation logic used by both form and chat paths."""
    inputs = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        return_dict=True,
        add_generation_prompt=True,
    ).to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=500,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

    input_length = inputs["input_ids"].shape[1]
    new_tokens = out[0][input_length:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def form_advise(location, crop, observation, question):
    """Single-shot advisor using structured form input."""
    if not all([location, crop, observation, question]):
        return "Please fill in all four fields so the advisor can give useful guidance."

    user_msg = f"""A farmer needs advice.

Location/region: {location}
Current or planned crop: {crop}
Recent weather/climate observation: {observation}
Specific question or concern: {question}

Give your advice now."""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]
    return _generate(messages)


def chat_advise(message, history):
    """Multi-turn chat for follow-up conversations."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": message})
    return _generate(messages)


# ----- HANDOFF: form → chat -----

def handoff_to_chat(location, crop, observation, question, form_response):
    """Package the form Q&A as the opening of a chat conversation."""
    if not form_response or "Please fill in all four fields" in form_response:
        return [], gr.Tabs(selected="form_tab")

    user_display = (
        f"**My situation:**\n"
        f"- Region: {location}\n"
        f"- Crop: {crop}\n"
        f"- Observation: {observation}\n\n"
        f"**My question:** {question}"
    )

    history = [
        {"role": "user", "content": user_display},
        {"role": "assistant", "content": form_response},
    ]
    return history, gr.Tabs(selected="chat_tab")


def chat_respond(user_message, chat_history):
    if not user_message.strip():
        return "", chat_history
    chat_history = chat_history + [{"role": "user", "content": user_message}]
    bot_reply = chat_advise(user_message, chat_history[:-1])
    chat_history = chat_history + [{"role": "assistant", "content": bot_reply}]
    return "", chat_history


# ----- UI -----

with gr.Blocks(title="AgriClima — Climate-Resilient Farming Advisor") as demo:
    gr.Markdown(
        """
        # 🌱 AgriClima
        ### A climate-resilient farming advisor for smallholder farmers
        Powered by Gemma 4 — designed to work in low-connectivity settings.
        """
    )

    with gr.Tabs() as tabs:
        # ===== TAB 1: FORM =====
        with gr.Tab("Quick consultation", id="form_tab"):
            gr.Markdown("Fill in all four fields. The advisor gives best results with specific input.")

            with gr.Row():
                with gr.Column():
                    f_location = gr.Textbox(
                        label="Your region",
                        placeholder="e.g., Punjab, India  •  Western Kenya  •  Northern Ghana",
                    )
                    f_crop = gr.Textbox(
                        label="Crop you grow / plan to grow",
                        placeholder="e.g., wheat, maize, sorghum, cassava",
                    )
                    f_observation = gr.Textbox(
                        label="What you've noticed about the weather/climate recently",
                        placeholder="e.g., Rains came one month later than normal. Days are hotter.",
                        lines=3,
                    )
                    f_question = gr.Textbox(
                        label="Your specific question or concern",
                        placeholder="e.g., Should I still plant wheat this season?",
                        lines=2,
                    )
                    f_submit = gr.Button("Get advice", variant="primary")

                with gr.Column():
                    f_output = gr.Markdown(label="AgriClima says:")
                    f_continue = gr.Button(
                        "💬 Continue in conversation",
                        variant="secondary",
                    )

            gr.Examples(
                examples=[
                    ["Punjab, India", "wheat",
                     "Winter is 4°C warmer than the 10-year average. Soil is unusually dry.",
                     "Should I still plant wheat next season or switch?"],
                    ["Western Kenya", "maize",
                     "The long rains have been arriving 3 weeks later than usual for two years. Some neighbors lost their crop last season.",
                     "When should I plant this season, and should I be doing anything different?"],
                    ["Northern Ghana", "cassava (4 months into growth)",
                     "An unusual heatwave has temperatures 8 degrees above normal for 10 days. Leaves are wilting visibly.",
                     "Is my crop in immediate danger and what can I do this week to save it?"],
                    ["central India", "vegetables",
                     "weather has been unpredictable",
                     "what should I grow next season?"],
                ],
                inputs=[f_location, f_crop, f_observation, f_question],
                label="Try one of these example questions",
            )

        # ===== TAB 2: CHAT =====
        with gr.Tab("Continue conversation", id="chat_tab"):
            gr.Markdown(
                "Use this tab to ask follow-up questions after starting a consultation in the **Quick consultation** tab. "
                "You can also start fresh here, but specific input gives better results."
            )

            chatbot = gr.Chatbot(
                type="messages",
                height=450,
                show_copy_button=True,
                avatar_images=(None, "🌱"),
            )

            chat_msg = gr.Textbox(
                label="Your message",
                placeholder="Ask a follow-up question, or describe a new situation in detail (location, crop, observation, question).",
                lines=3,
            )

            with gr.Row():
                send_btn = gr.Button("Send", variant="primary", scale=4)
                clear_btn = gr.Button("Start new conversation", scale=1)

    gr.Markdown(
        "*AgriClima provides general guidance only. Always confirm major decisions "
        "with local agricultural extension officers when possible.*"
    )

    # ----- Wire up events -----
    f_submit.click(
        fn=form_advise,
        inputs=[f_location, f_crop, f_observation, f_question],
        outputs=f_output,
    )

    f_continue.click(
        fn=handoff_to_chat,
        inputs=[f_location, f_crop, f_observation, f_question, f_output],
        outputs=[chatbot, tabs],
    )

    send_btn.click(chat_respond, [chat_msg, chatbot], [chat_msg, chatbot])
    chat_msg.submit(chat_respond, [chat_msg, chatbot], [chat_msg, chatbot])
    clear_btn.click(lambda: [], None, chatbot)


if __name__ == "__main__":
    demo.launch()
