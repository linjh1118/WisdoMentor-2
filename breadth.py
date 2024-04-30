base_instruction = "I want you act as a Prompt Creator 并用中文生成所有内容,\r\n\
Your goal is to draw inspiration from the #Given Prompt# to create a brand new prompt所生成回答应符合以下模板：\r\n\
	                我已经学过的前置内容如下:\r\n\
	                第一点：\r\n\
	                第二点：\r\n\
	                ......\r\n\
	                我的问题是：......\r\n\
This new prompt should belong to the same domain as the #Given Prompt# but be even more rare.“我已经学过的前置内容如下”中的内容不仅限于两点，可扩充\r\n\
The LENGTH and complexity of the #Created Prompt# should be similar to that of the #Given Prompt#.\r\n\
The #Created Prompt# must be reasonable and must be understood and responded by humans.#Created Prompt# should realted to the knowledge offered by #The Given Prompt#,and also contain the knowledge required to solve the problem in #Created Prompt# and not being included in the #The Given Prompt#.\r\n\
'#Given Prompt#', '#Created Prompt#', 'given prompt' and 'created prompt' are not allowed to appear in #Created Prompt#,if you didn't know the answer,just answer 'Sorry, I don't know.'.\r\n"



def createBreadthPrompt(instruction):
	prompt = base_instruction
	prompt += "#Given Prompt#: \r\n {} \r\n".format(instruction)
	prompt += "#Created Prompt#:\r\n"
	return prompt
