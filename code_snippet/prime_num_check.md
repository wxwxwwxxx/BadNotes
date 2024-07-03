判断一个数字是否是质数的方法有很多种，以下是几种常见的方法，从简单到复杂：

### 1. 试除法 (Trial Division)
这种方法最简单，但对较大的数字效率较低。

**步骤：**
1. 如果 \( n \) 小于 2，则 \( n \) 不是质数。
2. 检查 \( n \) 是否能被 2 整除，如果能，则 \( n \) 不是质数。
3. 从 3 开始，检查所有小于或等于 \( \sqrt{n} \) 的奇数，如果 \( n \) 能被任何一个奇数整除，则 \( n \) 不是质数。

**Python 示例：**
```python
import math

def is_prime(n):
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    for i in range(5, int(math.sqrt(n)) + 1, 6):
        if n % i == 0 or n % (i + 2) == 0:
            return False
    return True
```

### 2. 朴素试除法优化版 (Optimized Trial Division)
这是试除法的一种优化版本，利用了6k ± 1的特性。

**步骤：**
1. 如果 \( n \) 小于 2，则 \( n \) 不是质数。
2. 检查 \( n \) 是否能被 2 或 3 整除，如果能，则 \( n \) 不是质数。
3. 从 5 开始，以 6 为步长检查 \( n \) 是否能被 \( 6k ± 1 \) 整除。

**Python 示例：**
```python
import math

def is_prime(n):
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True
```

### 3. 埃拉托斯特尼筛法 (Sieve of Eratosthenes)
适用于需要判断多个数字是否为质数的情况。

**步骤：**
1. 创建一个大小为 \( n \) 的布尔数组，将所有元素初始化为 True。
2. 从 2 开始，标记所有 2 的倍数、3 的倍数、……，直到 \( \sqrt{n} \) 的倍数。
3. 剩下的所有未标记的数字都是质数。

**Python 示例：**
```python
def sieve_of_eratosthenes(n):
    is_prime = [True] * (n + 1)
    p = 2
    while (p * p <= n):
        if is_prime[p]:
            for i in range(p * p, n + 1, p):
                is_prime[i] = False
        p += 1
    prime_numbers = [p for p in range(2, n + 1) if is_prime[p]]
    return prime_numbers
```

### 4. Miller-Rabin 素性测试 (Miller-Rabin Primality Test)
适用于大数字的快速质数判断，特别是在对可能的质数进行概率判断时。

**步骤：**
1. 将 \( n \) 分解为 \( d \cdot 2^r + 1 \) 形式。
2. 随机选择多个基数 \( a \)，进行幂次方模运算检查 \( n \) 是否为合数。
3. 通过多次测试来减少错误率。

**Python 示例：**
```python
import random

def miller_rabin(n, k=5):  # number of tests
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0:
        return False

    # Write n as d*2^r + 1
    r, d = 0, n - 1
    while d % 2 == 0:
        d //= 2
        r += 1

    # Witness loop
    for _ in range(k):
        a = random.randint(2, n - 2)
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(r - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    return True
```

### 5. AKS 素数测试 (AKS Primality Test)
这种方法是确定性的，可以在多项式时间内判断一个数是否是质数，但由于其复杂性和实际性能，不常用于实战。

对于一般用途，推荐使用 **优化试除法** 或 **Miller-Rabin** 进行单个数字的质数判断，对于大范围的数字，可以使用 **埃拉托斯特尼筛法**。