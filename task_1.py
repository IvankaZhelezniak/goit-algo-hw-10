
from typing import Dict, List

DEFAULT_COINS = [50, 25, 10, 5, 2, 1]

# Функція яка приймає суму amount (яку потрібно видати покупцеві) і список номіналів coins
# і повертає словник {номінал: кількість} із кількістю монет кожного номіналу, 
# що використовуються для формування цієї суми,  або кидає ValueError, якщо суму неможливо набрати.
def find_coins_greedy(amount: int, coins: List[int] = None) -> Dict[int, int]:
    """
    Жадібний алгоритм видачі решти.
    Повертає словник {номінал: кількість}, використовуючи спочатку найбільші монети.
    """
    if coins is None:
        coins = DEFAULT_COINS
    if amount < 0:
        raise ValueError("amount must be non-negative")
    result: Dict[int, int] = {}
    remaining = amount
    for c in sorted(coins, reverse=True):
        if c <= 0:
            continue
        count, remaining = divmod(remaining, c)
        if count:
            result[c] = count
        if remaining == 0:
            break
    if remaining != 0:
        raise ValueError(f"Суму {amount} неможливо набрати заданими монетами {coins}")
    return result

def find_min_coins(amount: int, coins: List[int] = None) -> Dict[int, int]:
    """
    Динамічне програмування (Bottom-Up): мінімальна кількість монет для amount.
    Повертає словник {номінал: кількість}. Якщо суму неможливо набрати — кидає ValueError.
    Часова складність: O(amount * k), де k = кількість номіналів.
    Пам'ять: O(amount).
    """
    if coins is None:
        coins = DEFAULT_COINS
    if amount < 0:
        raise ValueError("amount must be non-negative")
    if amount == 0:
        return {}
    INF = 10**18
    min_coins = [0] + [INF] * amount
    last_coin = [-1] * (amount + 1)
    for a in range(1, amount + 1):
        best = INF
        best_coin = -1
        for c in coins:
            if c <= 0:
                continue
            if a - c >= 0 and min_coins[a - c] + 1 < best:
                best = min_coins[a - c] + 1
                best_coin = c
        min_coins[a] = best
        last_coin[a] = best_coin
    if min_coins[amount] >= INF:
        raise ValueError(f"Суму {amount} неможливо набрати заданими монетами {coins}")
    res: Dict[int, int] = {}
    a = amount
    while a > 0:
        c = last_coin[a]
        if c == -1:
            raise RuntimeError("Відновлення рішення не вдалося")
        res[c] = res.get(c, 0) + 1
        a -= c
    return dict(sorted(res.items()))

def _time_func(func, *args, repeats: int = 5, **kwargs) -> float:
    start = __import__('time').perf_counter
    total = 0.0
    for _ in range(repeats):
        t0 = start()
        func(*args, **kwargs)
        total += (start() - t0)
    return total / repeats

if __name__ == "__main__":
    tests = [113, 9999, 123456]
    for amount in tests:
        g = find_coins_greedy(amount)
        d = find_min_coins(amount)
        print(f"Сума: {amount}")
        print("  Greedy :", g)
        print("  DP     :", d)
        tg = _time_func(find_coins_greedy, amount)
        td = _time_func(find_min_coins, amount)
        print(f"  Час (Greedy) ~ {tg:.6f}s  |  Час (DP) ~ {td:.6f}s")
        print("-" * 60)

    print(find_coins_greedy(113))
    print(find_min_coins(113))