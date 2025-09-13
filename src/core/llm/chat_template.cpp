#include "chat_template.h"
#include <algorithm>
#include <sstream>

namespace rtc_runner {

// Llama 3.2 chat template tokens
const std::string ChatTemplate::BOS_TOKEN = "<|begin_of_text|>";
const std::string ChatTemplate::EOS_TOKEN = "<|end_of_text|>";
const std::string ChatTemplate::SYSTEM_START =
    "<|start_header_id|>system<|end_header_id|>\n\n";
const std::string ChatTemplate::SYSTEM_END = "<|eot_id|>";
const std::string ChatTemplate::USER_START =
    "<|start_header_id|>user<|end_header_id|>\n\n";
const std::string ChatTemplate::USER_END = "<|eot_id|>";
const std::string ChatTemplate::ASSISTANT_START =
    "<|start_header_id|>assistant<|end_header_id|>\n\n";
const std::string ChatTemplate::ASSISTANT_END = "<|eot_id|>";

std::string ChatTemplate::format_message(const ChatMessage &message) {
  std::string formatted;

  switch (message.role) {
  case ChatMessage::SYSTEM:
    formatted = SYSTEM_START + trim(message.content) + SYSTEM_END;
    break;
  case ChatMessage::USER:
    formatted = USER_START + trim(message.content) + USER_END;
    break;
  case ChatMessage::ASSISTANT:
    formatted = ASSISTANT_START + trim(message.content) + ASSISTANT_END;
    break;
  }

  return formatted;
}

std::string
ChatTemplate::format_conversation(const std::vector<ChatMessage> &messages) {
  if (messages.empty()) {
    return "";
  }

  std::ostringstream result;
  result << BOS_TOKEN;

  for (const auto &message : messages) {
    result << format_message(message);
  }

  // Add assistant start for generation (don't close it)
  result << ASSISTANT_START;

  return result.str();
}

std::string ChatTemplate::format_user_prompt(const std::string &prompt,
                                             const std::string &system_prompt) {
  std::vector<ChatMessage> messages;

  // Add system prompt if provided
  if (!system_prompt.empty()) {
    messages.emplace_back(ChatMessage::SYSTEM, system_prompt);
  }

  // Add user message
  messages.emplace_back(ChatMessage::USER, prompt);

  return format_conversation(messages);
}

std::string ChatTemplate::role_to_string(ChatMessage::Role role) {
  switch (role) {
  case ChatMessage::SYSTEM:
    return "system";
  case ChatMessage::USER:
    return "user";
  case ChatMessage::ASSISTANT:
    return "assistant";
  default:
    return "unknown";
  }
}

std::string ChatTemplate::trim(const std::string &str) {
  size_t first = str.find_first_not_of(" \t\n\r");
  if (first == std::string::npos) {
    return "";
  }

  size_t last = str.find_last_not_of(" \t\n\r");
  return str.substr(first, (last - first + 1));
}

} // namespace rtc_runner