import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

model_id = "Qwen/Qwen2.5-1.5B-Instruct"
adapter_path = "./student_advisor_adapter"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Use the same 4-bit quantization config used during training
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float32,
    bnb_4bit_use_double_quant=True
)

print("Loading base model in 4-bit...")
base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto"
)

# Load your fine-tuned LoRA adapter on top
print("Loading fine-tuned student advisor adapter...")
model = PeftModel.from_pretrained(base_model, adapter_path)

# Your exact prompt and data structure
system_prompt = "You are an academic advisor for CSE II Year. Analyze the target student against the class database and output the assessment in strict raw JSON without any markdown formatting or extra text.\n\nReturn JSON with these exact keys:\n- \"student_name\"\n- \"overall_standing\"\n- \"class_rank\"\n- \"positives\" (array)\n- \"negatives\" (array)\n- \"suggestions_to_improve\" (array)"

user_input = "Data:\nTarget Student: Alice Smith. Database: [{'student_name': 'Alice Smith', 'grades': [{'course_name': 'Data Structures', 'marks_obtained': 88, 'max_marks': 100}]}, {'student_name': 'Bob Jones', 'grades': [{'course_name': 'Data Structures', 'marks_obtained': 62, 'max_marks': 100}]}]"

# Format prompt using Qwen chat template
prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_input}<|im_end|>\n<|im_start|>assistant\n"

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

print("\n--- GENERATING RESPONSE FROM FINE-TUNED ADVISOR ---")
outputs = model.generate(
    **inputs, 
    max_new_tokens=300, 
    temperature=0.1,  # Low temperature for strict, consistent JSON compliance
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id
)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
assistant_response = response.split("<|im_start|>assistant\n")[-1]
print(assistant_response)