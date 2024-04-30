base_instruction = "I want you act as a Prompt Rewriter and rewrite prompts in Chinese，并用中文生成所有内容。你的回答应符合以下模板\r\n\
	                前置内容:\r\n\
                    第一点：\r\n\
                    第二点：\r\n\
                    ......\r\n\
					回答：......\r\n\\r\n\
					Your objective is to answer the  given prompt，在“前置内容”里列出解决该问题需要的知识并详细介绍其内容，不仅限于两点，可以扩充\r\n \
					Your rewriting cannot omit the non-text parts such as the table and code in #The Given Prompt#:. Also, please do not omit the input in #The Given Prompt#. \r\n \
					You SHOULD complicate the given prompt using the following method: \r\n\
					{} \r\n\
					You should try your best not to make the #answer# become verbose.\r\n\
					'#The Given Prompt#', '#Rewritten Prompt#', 'given prompt' and 'rewritten prompt' are not allowed to appear in #answer#,if you didn't know the answer,just answer '我不知道'.\r\n"

def answer(instruction):
	prompt = base_instruction
	prompt += "#Given Prompt#: \r\n {} \r\n".format(instruction)
	return prompt