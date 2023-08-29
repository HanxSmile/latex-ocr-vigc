import gradio as gr
import torch
import argparse
from vigc.common.config import Config
import vigc.tasks as tasks
from vigc.processors import BlipImageEvalProcessor


def parse_args():
    parser = argparse.ArgumentParser(description="Training")

    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
             "in xxx=yyy format will be merged into config file (deprecate), "
             "change to --cfg-options instead.",
    )

    args = parser.parse_args()
    # if 'LOCAL_RANK' not in os.environ:
    #     os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


if __name__ == '__main__':
    cfg = Config(parse_args())
    task = tasks.setup_task(cfg)

    print('Loading model...')
    device = torch.device(f"cuda:{cfg.run_cfg.device}") if torch.cuda.is_available() else "cpu"
    model = task.build_model(cfg).to(device)
    vis_processors = BlipImageEvalProcessor(image_size=224)

    print('Loading model done!')

    image_input = gr.Image(type="pil")

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

    sampling = gr.Radio(
        choices=["Beam search", "Nucleus sampling"],
        value="Beam search",
        label="Text Decoding Method",
        interactive=True,
    )

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

    prompt_textbox = gr.Textbox(label="Prompt:", placeholder="prompt", lines=2)


    def inference(image, prompt, min_len, max_len, beam_size, len_penalty, repetition_penalty, top_p, decoding_method,
                  modeltype):
        use_nucleus_sampling = decoding_method == "Nucleus sampling"
        print(image, prompt, min_len, max_len, beam_size, len_penalty, repetition_penalty, top_p, use_nucleus_sampling)
        image = vis_processors(image).unsqueeze(0).to(device)

        samples = {
            "image": image,
            "prompt": [prompt],
        }

        output = model.caption_generate(
            samples,
            length_penalty=float(len_penalty),
            repetition_penalty=float(repetition_penalty),
            num_beams=beam_size,
            max_length=max_len,
            min_length=min_len,
            top_p=top_p,
            use_nucleus_sampling=use_nucleus_sampling,
        )

        return output[0]


    gr.Interface(
        fn=inference,
        inputs=[image_input, prompt_textbox, min_len, max_len, beam_size, len_penalty, repetition_penalty, top_p,
                sampling],
        outputs="text",
        allow_flagging="never",
    ).launch(share=True, enable_queue=True, server_name="0.0.0.0", debug=True)
