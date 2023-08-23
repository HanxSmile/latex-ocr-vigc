import gradio as gr
from vigc.common.demo_tools import inference, parse_arguments, prepare_models

if __name__ == '__main__':
    args = parse_arguments()
    all_models = prepare_models(args)
    inference = inference(all_models)

    with gr.Blocks() as demo:
        empty_text_box = gr.Textbox(visible=False)
        with gr.Row().style(equal_height=False):
            with gr.Column():
                model_type = gr.Radio(
                    choices=["MiniGPT4", "InstructBlip"],
                    value="InstructBlip",
                    label="Model Type",
                    interactive=True,
                )
                image_input = gr.Image(type="pil")
                with gr.Row():
                    min_len = gr.Slider(
                        minimum=1,
                        maximum=50,
                        value=1,
                        step=1,
                        interactive=True,
                        label="Min Length",
                    )

                    max_len = gr.Slider(
                        minimum=10,
                        maximum=500,
                        value=250,
                        step=5,
                        interactive=True,
                        label="Max Length",
                    )
                with gr.Row():
                    top_p = gr.Slider(
                        minimum=0.5,
                        maximum=1.0,
                        value=0.9,
                        step=0.1,
                        interactive=True,
                        label="Top p",
                    )

                    beam_size = gr.Slider(
                        minimum=1,
                        maximum=10,
                        value=5,
                        step=1,
                        interactive=True,
                        label="Beam Size",
                    )
                with gr.Row():
                    len_penalty = gr.Slider(
                        minimum=-1,
                        maximum=2,
                        value=1,
                        step=0.2,
                        interactive=True,
                        label="Length Penalty",
                    )

                    repetition_penalty = gr.Slider(
                        minimum=-1,
                        maximum=3,
                        value=1,
                        step=0.2,
                        interactive=True,
                        label="Repetition Penalty",
                    )

                with gr.Row():
                    last_infer_all = gr.Radio(
                        choices=["Truncation", "No Truncation"],
                        value="No Truncation",
                        label="Whether to Truncate the Answer",
                        interactive=True
                    )

                    in_section = gr.Radio(
                        choices=["In Paragraph", "In Sentence"],
                        value="In Paragraph",
                        label="Generate Style",
                        interactive=True
                    )
                with gr.Row().style(equal_height=True):
                    sampling = gr.Radio(
                        choices=["Beam search", "Nucleus sampling"],
                        value="Beam search",
                        label="Text Decoding Method",
                        interactive=True,
                    )

                    answer_length = gr.Slider(
                        minimum=1,
                        maximum=10,
                        value=4,
                        step=1,
                        interactive=True,
                        label="Answer Length"
                    )
            with gr.Column():
                with gr.Column():
                    task = gr.Radio(
                        choices=["complex reasoning", "conversation", "detail description"],
                        value="conversation",
                        label="Task",
                        interactive=True,
                    )
                    gen_qa_button = gr.Button("Generate QA-pairs", variant="primary", size="sm")

                with gr.Column():
                    question_textbox = gr.Textbox(label="Question:", placeholder="question", lines=2)
                    gen_ans_button = gr.Button("Generate Answer", variant="primary", size="sm")

                text_output = gr.Textbox(label="Output:")
            gen_qa_button.click(
                fn=inference,
                inputs=[image_input, empty_text_box, task, min_len, max_len, beam_size, len_penalty, repetition_penalty,
                        top_p, sampling, answer_length, last_infer_all, in_section, model_type],
                outputs=text_output
            )

            gen_ans_button.click(
                fn=inference,
                inputs=[image_input, question_textbox, task, min_len, max_len, beam_size, len_penalty,
                        repetition_penalty,
                        top_p, sampling, answer_length, last_infer_all, in_section, model_type],
                outputs=text_output
            )

    demo.launch(share=True, enable_queue=True, server_name="0.0.0.0", debug=True)
