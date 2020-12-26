import torch
import json 
from transformers import T5Tokenizer, T5ForConditionalGeneration

class T5_Tiny:

    def __init__(self, path, device):

        self.model = T5ForConditionalGeneration.from_pretrained(path)
        self.tokenizer = T5Tokenizer.from_pretrained(path)
        self.device = torch.device(device)

    def preProcess(self, data):

        preprocess_text = data.strip().replace("\n","").replace("\'s", "")
        return preprocess_text

    def getSummary(self, data):
        
        data = self.preProcess(data)
        t5_prepared_data = "summarize: "+data
        tokenized_text = self.tokenizer.encode(t5_prepared_data, return_tensors="pt").to(self.device)

        summary_ids = self.model.generate(
            tokenized_text, 
            temperature=0.6,
            num_beams=5, 
            no_repeat_ngram_size=2, 
            max_length=200)

        output = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return output

if __name__=="__main__":
    
    text = """
    The amount of physical memory in a handheld depends on the device, but typically it is somewhere between 1 MB and 1 GB. (Contrast this with a typical PC or workstation, which may have several gigabytes of memory.) As a result, the operating system and applications must manage memory efficiently. This includes returning all allocated memory to the memory manager when the memory is not being used. In Chapter 9, we explore virtual memory, which allows developers to write programs that behave as if the system has more memory than is physically available. Currently, not many handheld devices use virtual memory techniques, so program developers must work within the confines of limited physical memory.
    """
    tiny = T5_Tiny("t5_tiny", "cpu")
    print(tiny.getSummary(text))
