from reward_model import train_reward_model
from rl_algorithms import supervised_finetune, ppo_train, grpo_train, dpo_train


def main():
    print("=== Step 1: Train reward model ===")
    train_reward_model()

    print("=== Step 2: Supervised fine tune base policy ===")
    supervised_finetune()

    print("=== Step 3: PPO training ===")
    ppo_train()

    print("=== Step 4: GRPO training ===")
    grpo_train()

    print("=== Step 5: DPO training ===")
    dpo_train()


if __name__ == "__main__":
    main()
