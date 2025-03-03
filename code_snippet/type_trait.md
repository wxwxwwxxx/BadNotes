下面是一个简单的 `type_trait` 示例，用于判断一个类型是否是 `void`。它通过模板特化（template specialization）实现编译时的类型属性检查：

```cpp
#include <iostream>

// 主模板：默认情况下，类型不是void
template <typename T>
struct is_void {
    static constexpr bool value = false;
};

// 模板特化：当类型是void时，value为true
template <>
struct is_void<void> {
    static constexpr bool value = true;
};

int main() {
    std::cout << std::boolalpha;
    std::cout << is_void<int>::value << "\n";    // 输出 false
    std::cout << is_void<void>::value << "\n";   // 输出 true
    return 0;
}
```

### 核心原理：
1. **主模板**（Primary Template）定义了默认行为：`is_void<T>::value` 默认为 `false`。
2. **模板特化**（Template Specialization）针对特定类型（这里是 `void`）提供定制行为：`is_void<void>::value` 被设为 `true`。

### 扩展理解：
- `type_trait` 的本质是通过模板元编程（Template Metaprogramming）在编译时推导类型的属性。
- 类似的逻辑可以实现其他 `type_trait`，例如判断类型是否为指针、是否为整数等。

### 另一个例子：判断指针
```cpp
// 主模板：默认不是指针
template <typename T>
struct is_pointer {
    static constexpr bool value = false;
};

// 特化所有指针类型（如 int*, double* 等）
template <typename T>
struct is_pointer<T*> {
    static constexpr bool value = true;
};

// 使用示例：
std::cout << is_pointer<int>::value;    // 输出 false
std::cout << is_pointer<int*>::value;   // 输出 true
```

通过模板特化，`type_trait` 在编译期就能确定类型的属性，这是 C++ 元编程的核心机制之一。