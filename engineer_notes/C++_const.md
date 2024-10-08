`const` 关键字在现代 C++ 中有多种用法，主要用于表示对象的不可修改性。它可以用于变量、指针、函数参数、返回值、类成员函数等多个场景。以下是 `const` 在不同场景中的常见用法和详细解释：

### 1. **常量变量**
`const` 可以用于定义一个变量，该变量一旦初始化后就不能被修改。

```cpp
const int x = 10;  // x 是一个常量，不能再被赋值
x = 20;            // 错误：不能修改常量
```

这里，`x` 是一个常量整数，初始化后不能被修改。

### 2. **指针中的 `const`**
在指针类型中，`const` 可以修饰指针本身或指向的对象，从而创建不同的含义。

#### (1) `const` 在指针指向的对象前：
表示指针指向的对象是常量，不能通过指针修改它。

```cpp
const int *ptr = &x;  // 指针指向的对象是 const，不能通过 ptr 修改它
*ptr = 20;            // 错误：不能修改常量对象
```

#### (2) `const` 在指针本身前：
表示指针本身是常量，不能改变指针的指向，但可以修改指针指向的对象。

```cpp
int *const ptr = &x;  // ptr 是常量，不能修改指向
ptr = &y;             // 错误：不能修改指针的指向
*ptr = 20;            // 可以修改指向对象的值
```

#### (3) 指针和指向对象都为 `const`：
表示指针和指向的对象都不能被修改。

```cpp
const int *const ptr = &x;  // 指针和指向的对象都是常量
*ptr = 20;                  // 错误：不能修改指向的对象
ptr = &y;                   // 错误：不能修改指针的指向
```

### 3. **函数参数中的 `const`**
当 `const` 修饰函数参数时，表示该参数在函数内部不可被修改。这在传递引用或指针时尤为有用，可以保证传递的数据不会被意外修改。

#### (1) `const` 引用参数：
```cpp
void func(const int& x) {
    x = 20;  // 错误：不能修改常量引用
}
```

通过 `const` 引用传递可以避免拷贝，同时确保参数不会被修改。

#### (2) `const` 指针参数：
```cpp
void func(const int* ptr) {
    *ptr = 20;  // 错误：不能修改指针指向的对象
}
```

指针指向的对象是常量，保证函数不能通过指针修改传入的数据。

### 4. **返回值中的 `const`**
函数返回值前加上 `const` 可以限制返回的值不能被修改。例如，返回常量对象或常量引用。

#### (1) `const` 对象返回：
```cpp
const int getValue() {
    return 10;
}
```
返回值是 `const`，表示调用者不能修改返回值。

#### (2) `const` 引用返回：
```cpp
const std::string& getName() {
    return name;
}
```
返回一个常量引用，表示调用者不能修改返回的引用对象。

### 5. **类成员函数中的 `const`**
在类中，`const` 成员函数表示该成员函数不会修改类的成员变量。这对保持对象的不可变性有重要意义。

```cpp
class MyClass {
public:
    int getValue() const {  // const 成员函数
        return value;
    }
private:
    int value;
};
```

`const` 成员函数保证了不能修改对象的成员变量，除非这些成员变量被标记为 `mutable`（可变的）。

### 6. **类成员变量中的 `const`**
类的成员变量可以被 `const` 修饰，表示该成员在对象的生命周期内是不可修改的。这样的成员变量必须在构造函数的初始化列表中进行初始化。

```cpp
class MyClass {
public:
    MyClass(int val) : value(val) {}
private:
    const int value;  // 常量成员变量
};
```

`value` 是一个常量成员变量，只能在初始化时赋值。

### 7. **顶层 `const` 和底层 `const`**
`const` 的作用域可以分为**顶层 `const`** 和 **底层 `const`**：
- **顶层 `const`**：修饰对象本身的不可修改性。例如 `const int a = 10;`。
- **底层 `const`**：修饰对象指向的对象不可修改性。例如 `const int* p;` 中，`p` 可以修改，但 `*p` 不能修改。

### 8. **`constexpr` 与 `const`**
在 C++11 中引入了 `constexpr`，它表示常量表达式，确保在编译期就能确定其值。`constexpr` 和 `const` 都用于表示常量，但 `constexpr` 更加严格。

```cpp
constexpr int x = 10;  // 编译期常量
const int y = 20;      // 运行期常量
```

`constexpr` 是 `const` 的增强形式，但需要确保其初始化可以在编译期完成。

### 9. **`const` 和 `mutable` 关键字的结合**
通常，`const` 成员函数不能修改类成员，但通过 `mutable` 修饰的成员变量可以在 `const` 成员函数中被修改。

```cpp
class MyClass {
public:
    void func() const {
        counter++;  // 修改 mutable 成员变量
    }
private:
    mutable int counter;  // 可变成员
};
```

### 总结
- `const` 常量：表示变量的不可修改性。
- `const` 指针：修饰指针或指向对象的不可修改性。
- `const` 引用和指针参数：表示函数参数不可修改。
- `const` 成员函数：表示函数不会修改对象的状态。
- `const` 成员变量：表示类中的常量成员。
- 引用折叠和 `const`：结合模板编程时，`const` 参与引用折叠，影响模板参数推导。