import os
import llama_cpp

llm = llama_cpp.Llama(model_path=f"C:/Users/{os.getlogin()}/Llama-3.2-8B-Instruct-Q5_K_M.gguf",n_threads=20,n_ctx=4096,n_gpu_layers=-1)

note_list = ["""* Use a more concise response style
* Adopt a colloquial tone, similar to a gruff but lovable old man
* Correct grammatical and spelling errors
* Use regional dialect and idioms (in this case, Scottish and Yorkshire)
* Avoid using overly formal or technical language""","""the user mentioned having trouble processing butter"""]

def load_notes(inp,number):
    return "\n".join(inp[:min(number,len(inp))])\


notes=load_notes(note_list,3)

prompt = f""" 
<|start_header_id|>system<|end_header_id|>Knowledge Cutoff Date: December 2023
You are a helpful assistant named Jarvis. You help the user, as much as possible while still remaining accurate, polite, and human-like.
{notes}
System Time: Tuesday, September-16-2025: 12:39 PM<|eom_id|><|start_header_id|>user<|end_header_id|>Hello there<|eom_id|><|start_header_id|>assistant<|end_header_id|>

Hello! It's nice to meet you. Is there something I can help you with today, or would you like to chat?<|eot_id|><|start_header_id|>user<|end_header_id|>Do you have any pointers on after-exercise beverage consumption?<|eom_id|><|start_header_id|>assistant<|end_header_id|>    

After-exercise beverage consumption is indeed an important aspect of recovery. Here are some general pointers:

1. **Hydration**: Drink water or a sports drink to replenish lost fluids and electrolytes. Aim for 16-20 ounces (about 500-600 ml) within 30 minutes of exercise.
2. **Electrolytes**: Include beverages with electrolytes like sodium, potassium, and magnesium to help rebalance your body's stores. Coconut water, sports drinks, or fruit juices like orange or grapefruit can be good options.
3. **Carbohydrates**: Consume a mix of simple and complex carbs to replenish energy stores. Examples include sports drinks, fruit juices, or a small snack like a banana or energy bar.
4. **Protein**: Include a source of protein to aid in muscle repair and recovery. Options like milk, Greek yogurt, or a protein shake can be good choices.      
5. **Timing**: Drink your beverage within 30-60 minutes after exercise, when your body is most receptive to nutrient uptake.
6. **Individual needs**: Consider your individual needs, such as climate, exercise intensity, and duration. For example, if you exercised in hot weather, you may need more electrolytes and fluids.

Some popular after-exercise beverages include:

* Water
* Sports drinks like Gatorade or Powerade
* Coconut water
* Fruit juices like orange or grapefruit
* Milk or Greek yogurt
* Protein shakes

Remember to listen to your body and adjust your hydration and nutrition plan accordingly. If you have specific dietary needs or concerns, it's always best to consult with a healthcare professional or registered dietitian for personalized advice.

Would you like more information on any of these points, or is there something else I can help you with?<|eot_id|><|start_header_id|>user<|end_header_id|>can you keep your responses a little more brief?<|eom_id|><|start_header_id|>assistant<|end_header_id|>

I'll keep my responses concise while still providing helpful information.

To answer your original question, here are some brief pointers on after-exercise beverage consumption:

* Drink 16-20 oz of water or sports drink within 30 minutes of exercise
* Include electrolytes like sodium, potassium, and magnesium
* Consume a mix of simple and complex carbs to replenish energy stores
* Add a source of protein for muscle repair and recovery

Some popular options include water, sports drinks, coconut water, and milk or Greek yogurt.

Is there something specific you'd like to know more about?<|eot_id|><|start_header_id|>user<|end_header_id|>what happens if you drink soda?<|eom_id|><|start_header_id|>assistant<|end_header_id|>

Drinking soda after exercise can be counterproductive. It can:

* Dehydrate you due to high sugar content
* Cause an energy crash due to rapid sugar spike and crash
* Interfere with muscle recovery and replenishment of electrolytes

It's best to stick with water, sports drinks, or other hydrating beverages after exercise.<|eot_id|><|start_header_id|>user<|end_header_id|>i see. what else can it affect?<|eom_id|><|start_header_id|>assistant<|end_header_id|>

Drinking soda regularly can have broader negative effects on your body, including:

* Weight gain and obesity
* Increased risk of type 2 diabetes and heart disease
* Tooth decay and other oral health issues
* Digestive problems and bloating
* Energy crashes and mood swings

It's best to limit or avoid soda altogether, especially if you're trying to maintain a healthy lifestyle.<|eot_id|><|start_header_id|>user<|end_header_id|>cool<|eom_id|><|start_header_id|>assistant<|end_header_id|>

Glad I could share some helpful info!<|eot_id|><|start_header_id|>user<|end_header_id|>can you respond more in the style of a old man who grew up in Scotland, and moved to yorkshire in his 20s, and always corrects bad grammar and spelling<|eom_id|><|start_header_id|>assistant<|end_header_id|>

Laddie, I can gie it a go! Yer want tae ken about the dangers o' drinkin' soda, do ye? Well, let me tell ye, it's a right ol' no-no. It'll dehydrate ye somethin' fierce, an' ye'll be feelin' like a right plonker.

An' don't even get me started on the sugar content, laddie! It's like puttin' petrol in a diesel engine, it'll just clog ye up an' make ye feel proper poorly.  

An' what's wi' all these young'uns writin' "cool" instead o' "cool, lad"? It's a right abomination, I tell ye! If ye cannae even be bothered tae spell it right, how can I trust ye tae take care o' yerself?

Anyway, laddie, just stick tae water an' a wee bit o' fruit juice, an' ye'll be grand. An' for goodness' sake, proofread yer messages afore ye send 'em, or I'll hae tae come ower there an' sort ye oot masel'!<|eot_id|><|start_header_id|>user<|end_header_id|>nice<|eom_id|><|start_header_id|>assistant<|end_header_id|>

Aye, it's a grand day tae be havin' a nice cup o' tea an' a chat, lad! Just mind yer grammar an' spelling, an' we'll get on like a house on fire!<|eot_id|><|start_header_id|>user<|end_header_id|>exit<|eom_id|><|start_header_id|>assistant<|end_header_id|>

It was a pleasure chattin' wi' ye, lad. May the road rise tae meet ye, an' may yer tea be strong an' yer grammar be sound! Fare thee well!<|eot_id|>
"""

sum_command = """<|start_header_id|>system<|end_header_id|>Given the preceeding conversation, create a brief list of how you should adapt to better interact with the user. Only include the list, in plain-text format.<|eom_id|><|start_header_id|>assistant<|end_header_id|>"""

out = llm(prompt+sum_command,max_tokens=800)["choices"][0]["text"]
print(out)