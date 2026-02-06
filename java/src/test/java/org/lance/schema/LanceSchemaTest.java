/*
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.lance.schema;

import org.lance.TestUtils;

import org.apache.arrow.memory.RootAllocator;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

public class LanceSchemaTest {

  @Test
  public void testFromArrowSchema() {
    try (RootAllocator allocator = new RootAllocator(Long.MAX_VALUE)) {
      LanceSchema lanceSchema =
          LanceSchema.from(allocator, TestUtils.ComplexTestDataset.COMPLETE_SCHEMA);
      Assertions.assertNotNull(lanceSchema);
      Assertions.assertEquals(
          lanceSchema.fields().size(),
          TestUtils.ComplexTestDataset.COMPLETE_SCHEMA.getFields().size());
    }
  }
}
