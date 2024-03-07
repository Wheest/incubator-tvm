#ifndef CODEGEN_C_DISK_DATA_H_
#define CODEGEN_C_DISK_DATA_H_

#include <tvm/tir/expr.h>
#include <tvm/tir/function.h>
#include <tvm/tir/op_attr_types.h>
#include <tvm/tir/stmt.h>
#include <tvm/tir/stmt_functor.h>

namespace tvm {
namespace codegen {

using namespace tir;

class CodeGenC;

class VisitorState {
 public:
  virtual ~VisitorState() = default;
  virtual void VisitStmt_(const LetStmtNode* op, CodeGenC* codegen) = 0;
  virtual void LoadArrays(const CallNode* op, CodeGenC* codegen) = 0;

 protected:
  // Map of code strings for loading each variable from disk
  std::unordered_map<std::string, std::string> load_var_code_;
};

class DefaultVisitorState : public VisitorState {
 public:
  void VisitStmt_(const LetStmtNode* op, CodeGenC* codegen) override;
  void LoadArrays(const CallNode* op, CodeGenC* codegen) override;
};

class MainFuncVisitorState : public VisitorState {
 public:
  void VisitStmt_(const LetStmtNode* op, CodeGenC* codegen) override;
  void LoadArrays(const CallNode* op, CodeGenC* codegen) override;
};

}  // namespace codegen
}  // namespace tvm

#endif  // CODEGEN_C_DISK_DATA_H_
