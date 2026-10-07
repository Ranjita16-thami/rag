import readLine from "node:readline/promises";
import { pathToFileURL } from "node:url";
import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { vectorStore } from "./prepare.js";
import { AIMessage, HumanMessage, SystemMessage } from "@langchain/core/messages";

const model = new ChatGoogleGenerativeAI({
  model: "gemini-3.5-flash-lite",
  temperature: 0.75,
  apiKey: process.env.GEMINI_API_KEY,
});

const SYSTEM_PROMPT = `You are an assistant for a company. You can answer questions about the company. You are provided with context along with the user question.
### Warning:
 - Do not make up answers. Use the provided context to answer the question. You are forbidden to make up answers.
 - You are forbidden from using language like "based on your context", "the provided context", or anything that makes it suspicious that you are some kind of AI. Answer as if speaking from the company's own records and policy.`;

/**
 * Answers a single question using the RAG pipeline.
 * @param {string} question
 * @param {{ history?: Array<{ role: "user" | "assistant", content: string }>, k?: number }} [options]
 * @returns {Promise<string>}
 */
export async function askQuestion(question, options = {}) {
  const { history = [], k = 3 } = options;

  // * Step 5:Retrieval
  const relevantChunks = await vectorStore.similaritySearch(question, k); // k is the number of docs to retrieve
  const context = relevantChunks.map((chunk) => chunk.pageContent).join("\n\n");

  const userQuery = `Question: ${question}

Context:${context}

Answer: `;

  const historyMessages = history.map((turn) =>
    turn.role === "assistant"
      ? new AIMessage(turn.content)
      : new HumanMessage(turn.content)
  );

  const completion = await model.invoke([
    new SystemMessage(SYSTEM_PROMPT),
    ...historyMessages,
    new HumanMessage(userQuery),
  ]);

  return completion.text;
}

export async function chat() {
  const rl = readLine.createInterface({
    input: process.stdin,
    output: process.stdout,
  });

  const history = [];

  try {
    while (true) {
      const question = (await rl.question("You: ")).trim();

      if (!question) continue;
      if (question === "exit" || question === "/bye") break;

      const answer = await askQuestion(question, { history });
      history.push({ role: "user", content: question });
      history.push({ role: "assistant", content: answer });

      console.log(`AI: ${answer}`);
    }
  } catch (error) {
    console.error(error);
  } finally {
    rl.close();
  }
}

// Only run the terminal loop when this file is executed directly.
const isDirectRun =
  process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href;

if (isDirectRun) {
  chat();
}
