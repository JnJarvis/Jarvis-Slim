import time
import os
import dotenv
from rich.console import Console
from llama_cpp.llama_chat_format import Llama3VisionAlphaChatHandler
from llama_cpp import Llama

console = Console()
CPUTHRESH = 50
GPUTHRESH = 50
envfile = '.env'
color = dotenv.get_key(envfile, 'assistant_color')
debug = eval(dotenv.get_key(envfile, 'verbose'))

llm = Llama(model_path=f"C:/Users/{os.getlogin()}/Llama-3.2-8B-Instruct-Q5_K_M.gguf",
            n_gpu_layers=int(dotenv.get_key(envfile, 'number_gpu_layers')),
            verbose=eval(dotenv.get_key(envfile, 'verbose_llamacpp')),  # i was right
            n_ctx=int(dotenv.get_key(envfile, 'context_length')),
            n_batch=int(dotenv.get_key(envfile, 'batch_size')),
            n_threads=int(dotenv.get_key(envfile, 'number_threads')),
            flash_attn=bool(dotenv.get_key(envfile, 'flash_attention'))
            )

if debug:console.print('[bold italic]Llm loaded[/bold italic]')

def load_notes(inp,number):
    return "\n".join(inp[:min(number,len(inp))])

class ChatBot:
    def __init__(self, intro=None):
        self.default_length = 800
        self.llm = llm
        self.plugins = {}
        notes = open('notes.txt', 'r').readlines()
        notes=load_notes(notes,len(notes))
        if not intro:self.intro = f"""<|start_header_id|>system<|end_header_id|>Knowledge Cutoff Date: December 2023
You are a helpful assistant named Jarvis. You help the user, as much as possible while still remaining accurate, polite, and human-like.

Some helpful notes to keep in mind while talking to the user:
{notes}

System Time: {time.strftime("%A, %B-%d-%Y: %I:%M %p",)}<|eom_id|>"""
        else:self.intro = intro
        self.initial_prompt_layout = self.intro
        self.update_prompt()
        self.turnend = "<|eot_id|>"
        self.mesend = "<|eom_id|>"
        self.all = ""
        self.syshead = "<|start_header_id|>system<|end_header_id|>"
        self.assisthead = "<|start_header_id|>assistant<|end_header_id|>"
        self.user = "<|start_header_id|>user<|end_header_id|>"
        self.previousbotmes = ""
        
    def update_prompt_time(self):
        self.intro = eval(f" f'''{self.initial_prompt_layout}''' ")

    def update_prompt(self):
        self.update_prompt_time()

    def load_plugin(self, plugin_name, plugin_instance):
        self.plugins[plugin_name] = plugin_instance

    def reset(self):
        self.all=''

    def handle_message(self,message, write_input_to_context=True, write_output_to_context=True, override_default_max_length=None,stream=False):
        # Basic functionality of the chatbot
        self.update_prompt_time()
        output = ''
        prompt = message
        if write_input_to_context:
            self.all += "<|start_header_id|>user<|end_header_id|>"+prompt+"<|eom_id|><|start_header_id|>assistant<|end_header_id|>"
        
        self.all = self.all.removesuffix("<|start_header_id|>assistant<|end_header_id|>")
        self.all = self.all.removesuffix("<|start_header_id|>assistant<|end_header_id|>")
        self.all+="<|start_header_id|>assistant<|end_header_id|>"
        
        ranks = []
        responses = []
        for plugin_name, plugin_instance in self.plugins.items():
            ch, rank = plugin_instance.can_handle(message)
            ranks.append(rank)
            if ch:
                response = plugin_instance.handle_message(message)
                responses.append(
                    [response, rank, plugin_name, plugin_instance])

        for i in responses:
            if i[0] is not None and ranks:
                res, rk, plgnm, plgnin = i[0], i[1], i[2], i[3]
                if (rk == max(ranks)) and max(ranks) > 0:
                    if plgnin.rank != -1:
                        if write_output_to_context:
                            self.all += res.strip()+self.turnend
                        return res
                    
        if not self.all.endswith("<|start_header_id|>assistant<|end_header_id|>"):self.all+="<|start_header_id|>assistant<|end_header_id|>"
        
        if override_default_max_length != None:
            length = override_default_max_length
        else:
            length = self.default_length

        if not stream:
            output = llm(self.intro+self.all, stop=[self.turnend], temperature=0.3, max_tokens=length)["choices"][0]["text"]
            self.previousbotmes = output
            
            if write_output_to_context:
                self.all += output+self.turnend

            return output
        else:
            output = ""
            for i in llm(self.intro+self.all, stop=[self.turnend], temperature=0.3, max_tokens=length,stream=True):
                output+=i["choices"][0]["text"]
                yield i["choices"][0]["text"]
            if write_output_to_context:
                self.all += output+self.turnend


#Not used here, but If you happen to want to add plugins, start a discussion or something and I will add it
class Plugin:
    def __init__(self):
        self.can_prompt='None'
        self.req_can = 0

    def can_handle(self, message):
        raise NotImplementedError("Subclasses must implement this method")

    def handle_message(self, message):
        raise NotImplementedError("Subclasses must implement this method")

class AutoNotePlugin(Plugin):
    def __init__(self, bot: ChatBot, data_dir="./", default_file="notes.txt"):
        self.llm = bot.llm
        self.bot = bot
        self.data_dir = data_dir
        self.default_file = default_file
        self.rank = 0
        self.color = "gold1"
        self.req_can = 2
        self.can_prompt = """{self.bot.syshead}You are a Python function that takes in a Bot's previous message and a User message and outputs one value only: either True or False.
Output True only when the User is asking the assistant to remember, note, save, jot down, or record information for later.
If the Bot asks for confirmation about saving a note and the User confirms, output True.
If unclear, output False.
Examples:
Previous Bot Message: "Would you like me to remember that preference?", User Message: "yes, please save that", Output: True.
Previous Bot Message: "How can I help?", User Message: "what is 2+2", Output: False.
{self.bot.mesend}Bot Message: "{self.bot.previousbotmes}", User Message: "{message}", Output:"""

    def can_handle(self, message: str, override_should: bool = False, override_val=(False, False)):
        if not override_should:
            should_1 = self._should_save_note(message)
            should_2 = self._should_save_note(message)
        else:
            should_1 = override_val[0]
            should_2 = override_val[1]

        should = bool(should_1 and should_2)
        self.rank = 5 if should else -3
        if debug:
            console.print(
                f"[bold {self.color}]AutoNote should-save checks: {should_1}, {should_2} -> {should}[/bold {self.color}]"
            )
        return should, self.rank

    def handle_message(self, message):
        note = self._extract_note(message)
        
        if note == "":
            self.rank = -1
            return None

        filename = self.default_file
        file_path = self._write_note(filename, note)
        if file_path is None:
            self.rank = -1
            return None

        self.rank = 0
        return

    def _llm_text(self, prompt, max_tokens=32, stop=None):
        if stop is None:
            stop = [self.bot.turnend, self.bot.mesend]
        return self.llm(prompt, max_tokens=max_tokens, stop=stop)["choices"][0]["text"]

    def _parse_bool(self, text):
        cleaned = text.strip().lower().replace(".", "").replace(",", "")
        if cleaned.startswith("true"):
            return True
        if cleaned.startswith("false"):
            return False
        if ("true" in cleaned) and ("false" not in cleaned):
            return True
        return False

    def _clean_note(self, text):
        cleaned = text.strip().strip('"').strip("'")
        cleaned = " ".join(cleaned.split())
        return cleaned

    def _should_save_note(self, message):
        prompt = f"""{self.bot.syshead}You are a Python function that takes in a Bot's previous message and a User message and outputs one value only: either True or False.
Output True only when the User is asking the assistant to remember, note, save, jot down, or record information for later.
If the Bot asks for confirmation about saving a note and the User confirms, output True.
If unclear, output False.
Examples:
Previous Bot Message: "Would you like me to remember that preference?", User Message: "yes, please save that", Output: True.
Previous Bot Message: "How can I help?", User Message: "what is 2+2", Output: False.
Previous Bot Message: "How can I help?", User Message: "Please remember that I am allergic to peanuts", Output: True
{self.bot.mesend}Bot Message: "{self.bot.previousbotmes}", User Message: "{message}", Output:"""
        text = self._llm_text(prompt, max_tokens=2, stop=["\n", ".", self.bot.turnend, self.bot.mesend])
        out = self._parse_bool(text)
        if debug:
            console.print(f"[bold {self.color}]AutoNote should-save raw: {text.strip()}[/bold {self.color}]")
        return out

    def _extract_note(self, message):
        prompt = f"""{self.bot.syshead}You are a Python function that takes in a Bot's previous message and a User message and outputs a single note string to save for later.
Only output the note text. Do not include labels, markdown, bullets, or quotes.
If there is nothing meaningful to save, output NONE.
Examples:
Previous Bot Message: "What should I remember?", User Message: "Please remember that I am allergic to peanuts", Output: user is allergic to peanuts
Previous Bot Message: "Anything else?", User Message: "thanks", Output: NONE
{self.bot.mesend}Bot Message: "{self.bot.previousbotmes}", User Message: "{message}", Output:"""
        text = self._llm_text(prompt, max_tokens=100, stop=["\n", self.bot.turnend, self.bot.mesend])
        note = self._clean_note(text)
        if note.lower() == "none":
            return ""
        if debug:
            console.print(f"[bold {self.color}]AutoNote extracted note: {note}[/bold {self.color}]")
        return note

    def _write_note(self, filename, note):
        try:
            if not filename: filename = self.default_file
            full_path = os.path.join(self.data_dir, filename)
            with open(full_path, "a", encoding="utf-8") as file:
                file.writelines([note + "\n"])
            if debug:
                console.print(f"[bold {self.color}]AutoNote wrote to: {full_path}[/bold {self.color}]")
            return full_path
        except Exception as e:
            if debug:
                console.print(f"[bold red]AutoNote write failed: {e}[/bold red]")
            return None

def force_terminate():
    """
    Forcefully terminates the program immediately.
    """
    console.print("[bold red]Forcefully terminating the program...[/bold red]")
    os._exit(1)

if __name__ == "__main__":
    console.print("[italic light_grey]Chat with the bot; type exit to quit[/]")
    bot = ChatBot()
    auto_note_plugin = AutoNotePlugin(bot)
    bot.load_plugin("AutoNotePlugin", auto_note_plugin)

    input_st = time.time()
    user_input=""
    while user_input.lower() != "exit":
        try:
            user_input = console.input("\n[deep_sky_blue3]>>>[/]")
            input_st = time.time()
            response = bot.handle_message(user_input,stream=True)
            for i in response:
                console.print(f"[{color}]{i}[/]",end="")
            console.print("\n")
            if debug:console.print(f"[bold khaki]total_time:{time.time()-input_st}[/]")

            bot.update_prompt()
            print(bot.intro+bot.all)
        except KeyboardInterrupt:
            print(bot.intro+bot.all)
            force_terminate()

        except Exception as e:
            console.print(f"[bold red] Exception: {e} [/bold red]")
            force_terminate()