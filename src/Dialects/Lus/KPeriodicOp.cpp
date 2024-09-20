#include "KPeriodicOp.h"

namespace mlir {
  namespace lus {
    void KperiodicOp::build(Builder &odsBuilder,
			    OperationState &odsState,
			    StringRef word) {
      llvm::Twine t(word);
      Attribute w = StringAttr::get(odsBuilder.getContext(),t);
      odsState.addAttribute(getWordAttrName(),w);
      Type b = IntegerType::get(odsBuilder.getContext(),1);
      odsState.addTypes(b);
    }

    LogicalResult KperiodicOp::verify() {
      return success();
    }

    ParseResult KperiodicOp::parse(OpAsmParser &parser, OperationState &result){
      auto builder = parser.getBuilder();
      std::string word;
      if (parser.parseOptionalString(&word))
	return failure();
      llvm::Twine t(word);
      Attribute w = StringAttr::get(builder.getContext(),t);
      result.addAttribute(getWordAttrName(),w);
      Type b = IntegerType::get(builder.getContext(),1);
      result.addTypes(b);
      return success();
    }

    void KperiodicOp::print(OpAsmPrinter &p) {
      p << " ";
      StringRef w = getWord();
      p << "\"" << w << "\"";
    }

    std::vector<bool> KperiodicOp::getPrefix() {
      StringRef str = getWord();
      int lparenPosition;
      for(lparenPosition=0;
	  (lparenPosition<str.size())&&(str[lparenPosition]!='(');
	  lparenPosition++) ;
      assert(lparenPosition != str.size()) ;
      std::vector<bool> prefix;
      for(int i=0;i<lparenPosition;i++){
	if(str[i]=='0') prefix.push_back(false) ;
	else if(str[i]=='1') prefix.push_back(true) ;
	else assert(false) ;
      }
      return prefix;
    }

    std::vector<bool> KperiodicOp::getPeriod() {
      StringRef str = getWord();
      int lparenPosition, rparenPosition ;
      for(lparenPosition=0;
	  (lparenPosition<str.size())&&(str[lparenPosition]!='(');
	  lparenPosition++) ;
      assert(lparenPosition != str.size()) ;
      for(rparenPosition=0;
	  (rparenPosition<str.size())&&(str[rparenPosition]!=')');
	  rparenPosition++) ;
      assert(rparenPosition != str.size()) ;
      std::vector<bool> period;
      for(int i=lparenPosition+1;i<rparenPosition;i++){
	if(str[i]=='0') period.push_back(false) ;
	else if(str[i]=='1') period.push_back(true) ;
	else assert(false) ;
      }
      return period;
    }
  }
}
