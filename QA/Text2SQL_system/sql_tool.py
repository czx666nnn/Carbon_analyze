OPENAI_API_KEY='your-openai-api-key'
from langchain import ChatOpenAI, LLMChain
from langchain.prompts import PromptTemplate


class SQLGenerator:
    def __init__(self, openai_api_key, db_schema):
        """
        初始化 SQL 生成器
        """
        self.db_schema = db_schema
        self.llm = ChatOpenAI(
            model_name="gpt-4o-mini",
            openai_api_key = openai_api_key,
            temperature = 0
        )
        self.prompt = PromptTemplate(
            input_variables=["user_query", "db_schema"],
            template="""
            你是一个SQL生成器，基于用户的自然语言查询生成对应的SQL语句。请确保生成的SQL语句是针对以下数据库模式的：

            {db_schema}

            用户查询: {user_query}

            生成的SQL语句:
            """
        )
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)

    def generate_sql(self, user_query):
        """
        根据用户查询生成SQL语句
        """
        sql = self.chain.invoke({
            "user_query": user_query,
            "db_schema": self.db_schema
        })
        # 清理生成的 SQL 语句，去掉多余的 markdown 或反引号
        sql = sql["text"].replace("```sql\n", "").replace("```", "").strip()  # 清理多余标记
        return sql