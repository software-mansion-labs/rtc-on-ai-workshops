// Chat template for Llama 3.2 model
#pragma once

#include <string>
#include <vector>

namespace rtc_runner {

struct ChatMessage {
  enum Role { SYSTEM, USER, ASSISTANT };

  Role role;
  std::string content;

  ChatMessage(Role r, const std::string &c) : role(r), content(c) {}
};

class ChatTemplate {
public:
  // Llama 3.2 chat template tokens
  static const std::string BOS_TOKEN;
  static const std::string EOS_TOKEN;
  static const std::string SYSTEM_START;
  static const std::string SYSTEM_END;
  static const std::string USER_START;
  static const std::string USER_END;
  static const std::string ASSISTANT_START;
  static const std::string ASSISTANT_END;

  // Format a single message
  static std::string format_message(const ChatMessage &message);

  // Format a conversation history
  static std::string
  format_conversation(const std::vector<ChatMessage> &messages);

  // Format a simple user prompt (most common use case)
  static std::string format_user_prompt(const std::string &prompt,
                                        const std::string &system_prompt = "");

  // Extract role string for logging
  static std::string role_to_string(ChatMessage::Role role);

private:
  static std::string trim(const std::string &str);
};

} // namespace rtc_runner